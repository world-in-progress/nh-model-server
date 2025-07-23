import os
import json
import c_two as cc
from icrms.isolution import ISolution
from icrms.isimulation import ISimulation
from src.nh_model_server.core.config import settings
from fastapi import APIRouter, BackgroundTasks, HTTPException
from src.nh_model_server.core.task import task_manager, TaskStatus
from src.nh_model_server.core.simulation import simulation_process_manager
from src.nh_model_server.schemas.model import (
    CloneActionRequest, SolutionCheckRequest, CloneEnvRequest, 
    BuildProcessGroupRequest, StartSimulationRequest, StopSimulationRequest, 
    PauseSimulationRequest, ResumeSimulationRequest,
    GetCompletedStepsRequest, GetStepResultRequest, 
    CheckStepReadyRequest, GetSimulationStatusRequest, AddHumanActionRequest
)
from src.nh_model_server.core.bootstrapping_treeger import BT

router = APIRouter(prefix='/model', tags=['model'])

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
            print(req.solution_address)
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

# 3. 获取action
@router.post('/clone_actions')
def clone_actions(req: CloneActionRequest):
    try:
        with cc.compo.runtime.connect_crm(req.solution_address, ISolution) as solution:
            actions = solution.get_human_actions()
            if not actions:
                return {"status": "success", "message": "No actions found"}
            action_dir = os.path.join(settings.RESOURCE_PATH, req.solution_name, 'actions')
            os.makedirs(action_dir, exist_ok=True)
            
            for action in actions:
                action_id = action.get('action_id')
                if not action_id:
                    continue
                action_file = os.path.join(action_dir, f"{action_id}.json")
                with open(action_file, "w", encoding="utf-8") as f:
                    json.dump(action, f, ensure_ascii=False, indent=4)
                    
        return {"status": "success", "message": f"Saved {len(actions)} actions"}
    except Exception as e:
        return {"status": "fail", "message": str(e)}

# 4. 构建进程组
@router.post('/build_process_group')
def build_process_group(req: BuildProcessGroupRequest):
    try:
        with cc.compo.runtime.connect_crm(req.solution_address, ISolution) as solution:
            env = solution.get_env()
        group_id = simulation_process_manager.build_process_group(req.solution_name, req.simulation_name, req.group_type, env)
        return {"result": "success", "group_id": group_id}
    except Exception as e:
        return {"result": "fail", "error": str(e)}

# 5. 启动模拟
@router.post('/start_simulation')
def start_simulation(req: StartSimulationRequest):
    ok = simulation_process_manager.start_simulation(
        req.solution_name, req.simulation_name
    )
    return {"result": "started" if ok else "already running"}

# 6. 结束模拟
@router.post('/stop_simulation')
def stop_simulation(req: StopSimulationRequest):
    print("Stopping simulation:", req.simulation_name)
    ok = simulation_process_manager.stop_simulation(
        req.solution_name, req.simulation_name
    )
    return {"result": "stopped" if ok else "not running"}

# 7. 暂停模拟
@router.post('/pause_simulation')
def pause_simulation(req: PauseSimulationRequest):
    ok = simulation_process_manager.pause_simulation(
        req.solution_name, req.simulation_name
    )
    return {"result": "paused" if ok else "not running"}

# 8. 恢复模拟
@router.post('/resume_simulation')
def resume_simulation(req: ResumeSimulationRequest):
    ok = simulation_process_manager.resume_simulation(
        req.solution_name, req.simulation_name, req.simulation_address
    )
    return {"result": "resumed" if ok else "not running"}

# 9. 获取已完成步骤列表
@router.post('/get_completed_steps')
def get_completed_steps(req: GetCompletedStepsRequest):
    try:
        node_key = f'root.simulations.{req.simulation_name}'
        with BT.instance.connect(node_key, ISimulation) as simulation:
            completed_steps = simulation.get_completed_steps()
            return {
                "result": "success", 
                "completed_steps": completed_steps,
                "count": len(completed_steps)
            }
    except Exception as e:
        return {"result": "fail", "error": str(e)}

# 10. 获取指定步骤的结果数据
@router.post('/get_step_result')
def get_step_result(req: GetStepResultRequest):
    try:
        node_key = f'root.simulations.{req.simulation_name}'
        with BT.instance.connect(node_key, ISimulation) as simulation:
            result = simulation.get_step_result(req.step)
            if result is not None:
                return {
                    "result": "success",
                    "data": result
                }
            else:
                return {
                    "result": "not_ready",
                    "message": f"Step {req.step} is not ready or already retrieved"
                }
    except Exception as e:
        return {"result": "fail", "error": str(e)}

# 11. 检查步骤是否就绪
@router.post('/check_step_ready')
def check_step_ready(req: CheckStepReadyRequest):
    try:
        node_key = f'root.simulations.{req.simulation_name}'
        with BT.instance.connect(node_key, ISimulation) as simulation:
            is_ready = simulation.check_step_ready(req.step)
            return {
                "result": "success",
                "step": req.step,
                "ready": is_ready
            }
    except Exception as e:
        return {"result": "fail", "error": str(e)}

# 12. 获取模拟状态
@router.post('/get_simulation_status')
def get_simulation_status(req: GetSimulationStatusRequest):
    try:
        node_key = f'root.simulations.{req.simulation_name}'
        with BT.instance.connect(node_key, ISimulation) as simulation:
            status = simulation.get_simulation_status()
            return {
                "result": "success",
                "status": status
            }
    except Exception as e:
        return {"result": "fail", "error": str(e)}

# 13. 添加人类干预行为
@router.post('/add_human_action')
def add_human_action(req: AddHumanActionRequest):
    try:
        from icrms.isimulation import HumanAction
        
        # 将字典转换为HumanAction对象
        action = HumanAction.model_validate(req.action)
        
        node_key = f'root.simulations.{req.simulation_name}'
        with BT.instance.connect(node_key, ISimulation) as simulation:
            result = simulation.add_human_action(req.step, action)
            return {
                "result": "success" if result.get("success", False) else "fail",
                "message": result.get("message", "Unknown error")
            }
    except Exception as e:
        return {"result": "fail", "error": str(e)}