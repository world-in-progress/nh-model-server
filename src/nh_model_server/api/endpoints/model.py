from fastapi import APIRouter, Body, BackgroundTasks, HTTPException
from src.nh_model_server.schemas.model import Signal, SignalType, SolutionCheckRequest, CloneEnvRequest, BuildProcessGroupRequest, StartSimulationRequest
from src.nh_model_server.core.simulation import simulation_process_manager
from src.nh_model_server.core.task import task_manager, TaskStatus
import c_two as cc
from icrms.isolution import ISolution
import os
from src.nh_model_server.core.config import settings
import time

router = APIRouter(prefix='/model', tags=['model'])

@router.post('/signal')
def signal(body: Signal=Body(..., description='The signal to send')):

    signal_type = body.signal_type
    solution_name = body.signal_value.solution_name
    simulation_name = body.signal_value.simulation_name
    solution_address = body.signal_value.solution_address
    simulation_address = body.signal_value.simulation_address
    
    if signal_type == SignalType.START:

        # 1、获取初始数据
        with cc.compo.runtime.connect_crm(solution_address, ISolution) as solution:
            solution_data = solution.get_solution_data()

        # 2、分配资源文件夹
        resource_path = os.path.join(settings.RESOURCE_PATH, solution_name, simulation_name)
        os.makedirs(resource_path, exist_ok=True)


        # 3、调用模型start，同时启动守护进程监控结果文件
        ok = simulation_process_manager.start(
            solution_name, simulation_name, solution_data, resource_path,
            simulation_address, body.signal_value.step
        )

        return {'message': 'START SUCCESS' if ok else 'ALREADY RUNNING'}
        # return {'message': 'START SUCCESS'}

    elif signal_type == SignalType.STOP:

        # 1、调用模型stop
        ok = simulation_process_manager.stop(solution_name, simulation_name)

        return {'message': 'STOP SUCCESS'}
    
    elif signal_type == SignalType.ROLLBACK:
        # 1、获取回滚至的step的result
        # 2、删除当前step之后的所有step
        # 3、调用模型start
        return {'message': 'ROLLBACK SUCCESS'}
    elif signal_type == SignalType.PAUSE:
        # 1、调用模型stop
        return {'message': 'PAUSE SUCCESS'}
    elif signal_type == SignalType.RESUME:
        # 1、获取actions
        # 2、应用actions
        # 3、调用模型start
        return {'message': 'RESUME SUCCESS'}
    else:
        return {'message': 'INVALID SIGNAL TYPE'}

# 1. 校验 solution
@router.post('/validate_solution')
def validate_solution(req: SolutionCheckRequest):
    solution = req.solution
    if 'type' not in solution:
        return {"valid": False, "message": "Missing 'type' in solution."}
    return {"valid": True, "message": "Solution is valid."}

# 2. 环境克隆
@router.post('/clone_env')
def clone_env(req: CloneEnvRequest, background_tasks: BackgroundTasks):
    task_id = task_manager.create_task(info={"solution_name": req.solution_name})
    def do_clone(task_id):
        try:
            task_manager.update_task(task_id, status=TaskStatus.RUNNING, progress=0)
            # Connect to remote ISolution and clone env
            with cc.compo.runtime.connect_crm(req.solution_address, ISolution) as solution:
                env_data = solution.clone_env()
            # Prepare resource directory
            resource_dir = os.path.join(settings.RESOURCE_PATH, req.solution_name)
            os.makedirs(resource_dir, exist_ok=True)
            total = len(env_data)
            for idx, (key, value) in enumerate(env_data.items()):
                file_name = value['file_name']
                file_path = os.path.join(resource_dir, file_name)
                content = value['content']
                # 判断 content 类型
                if isinstance(content, (bytes, bytearray)):
                    mode = 'wb'
                    open_kwargs = {}
                else:
                    mode = 'w'
                    open_kwargs = {'encoding': 'utf-8'}
                with open(file_path, mode, **open_kwargs) as f:
                    if mode == 'w':
                        # 如果是 list，转为字符串
                        if isinstance(content, list):
                            f.write(''.join(str(line) for line in content))
                        else:
                            f.write(str(content))
                    else:
                        f.write(content)
                # Update progress
                task_manager.update_task(task_id, progress=int((idx+1)/total*100))
            task_manager.update_task(task_id, status=TaskStatus.SUCCESS, progress=100)
        except Exception as e:
            task_manager.update_task(task_id, status=TaskStatus.FAILED, info={"error": str(e)})
    background_tasks.add_task(do_clone, task_id)
    return {"task_id": task_id}

# 获取克隆进度
@router.get('/clone_progress/{task_id}')
def clone_progress(task_id: str):
    progress = task_manager.get_task_progress(task_id)
    if progress == -1:
        raise HTTPException(404, detail="Task not found")
    return progress

# 获取所有任务
@router.get('/list_tasks')
def list_tasks():
    return task_manager.list_tasks()

# 3. 构建进程组
@router.post('/build_process_group')
def build_process_group(req: BuildProcessGroupRequest):
    try:
        config = simulation_process_manager.get_process_group_config(req.group_type)
        if not config:
            raise HTTPException(400, detail="Unknown group_type")
        # 组装每个进程的参数
        process_params = {}
        for proc in config["processes"]:
            name = proc["name"]
            args = req.process_args.get(name, [])
            process_params[name] = args
        group_id = simulation_process_manager.build_process_group(req.solution_name, req.simulation_name, req.group_type, process_params=process_params)
        return {"result": "success", "group_id": group_id}
    except Exception as e:
        return {"result": "fail", "error": str(e)}

# 4. 启动模拟
@router.post('/start_simulation')
def start_simulation(req: StartSimulationRequest):
    ok = simulation_process_manager.start(
        req.solution_name, req.simulation_name, req.solution_data, req.resource_path, req.simulation_address, req.step
    )
    return {"result": "started" if ok else "already running"}