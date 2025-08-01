import os
import zipfile
import io
import c_two as cc
from ...schemas.base import BaseResponse
from icrms.isolution import ISolution
from icrms.isimulation import ISimulation
from icrms.itreeger import ReuseAction, CRMDuration
from src.nh_model_server.core.config import settings
from fastapi import APIRouter, BackgroundTasks, HTTPException
from src.nh_model_server.core.task import task_manager, TaskStatus
from src.nh_model_server.core.simulation import simulation_process_manager
from src.nh_model_server.schemas.model import (
    SolutionCheckRequest, ClonePackageRequest,
    BuildProcessGroupRequest, StartSimulationRequest, StopSimulationRequest, 
    PauseSimulationRequest, ResumeSimulationRequest,
    GetCompletedStepsRequest, GetStepResultRequest, 
    CheckStepReadyRequest, GetSimulationStatusRequest, AddHumanActionRequest, GetStepResultResponse
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
@router.post('/clone_package', response_model=BaseResponse)
def clone_package(req: ClonePackageRequest, background_tasks: BackgroundTasks):
    try:
        task_id = task_manager.create_task(info={"solution_node_key": req.solution_node_key})
        def do_clone(task_id):
            try:
                task_manager.update_task(task_id, status=TaskStatus.RUNNING, progress=0)
                with cc.compo.runtime.connect_crm(req.solution_address, ISolution) as solution:
                    clone_result = solution.clone_package()
                clone_package_data = clone_result.get('package_data')
                
                if not clone_result.get('status', False) or not clone_package_data:
                    task_manager.update_task(task_id, status=TaskStatus.FAILED, info={"error": "Clone package failed or no package data"})
                    return
                
                resource_dir = os.path.join(settings.SOLUTION_DIR, req.solution_node_key)
                os.makedirs(resource_dir, exist_ok=True)
                
                # 如果 clone_package_data 是二进制数据（压缩包）
                if isinstance(clone_package_data, (bytes, bytearray)):
                    # 使用 zipfile 解压
                    with zipfile.ZipFile(io.BytesIO(clone_package_data), 'r') as zip_ref:
                        # 获取压缩包中的文件列表
                        file_list = zip_ref.namelist()
                        total_files = len(file_list)
                        
                        if total_files == 0:
                            task_manager.update_task(task_id, status=TaskStatus.FAILED, info={"error": "Empty zip package"})
                            return
                        
                        # 解压所有文件并更新进度
                        for idx, file_name in enumerate(file_list):
                            # 确保安全的文件路径，防止路径遍历攻击
                            safe_path = os.path.normpath(file_name)
                            if safe_path.startswith('..') or os.path.isabs(safe_path):
                                continue  # 跳过不安全的路径
                            
                            # 提取文件到目标目录
                            zip_ref.extract(file_name, resource_dir)
                            
                            # 更新进度
                            progress = int((idx + 1) / total_files * 100)
                            task_manager.update_task(task_id, progress=progress)
                        
                        task_manager.update_task(task_id, status=TaskStatus.SUCCESS, progress=100)
                else:
                    # 如果不是二进制数据，保持原有逻辑作为后备
                    task_manager.update_task(task_id, status=TaskStatus.FAILED, info={"error": "Package data is not in expected binary format"})
                    
            except zipfile.BadZipFile:
                task_manager.update_task(task_id, status=TaskStatus.FAILED, info={"error": "Invalid zip file format"})
            except Exception as e:
                task_manager.update_task(task_id, status=TaskStatus.FAILED, info={"error": str(e)})
        background_tasks.add_task(do_clone, task_id)
        return BaseResponse(success=True, message=task_id)
    except Exception as e:
        return BaseResponse(success=False, message=str(e))

# 获取克隆进度
@router.get('/clone_progress/{task_id}')
def clone_progress(task_id: str):
    progress = task_manager.get_task_progress(task_id)
    if progress == -1:
        raise HTTPException(404, detail="Task not found")
    return progress

# 3. 构建进程组
@router.post('/build_process_group', response_model=BaseResponse)
def build_process_group(req: BuildProcessGroupRequest):
    try:
        simulation_node_key = f'root.simulations.{req.simulation_name}'
        with cc.compo.runtime.connect_crm(req.solution_address, ISolution) as solution:
            model_env = solution.get_model_env()
        group_id = simulation_process_manager.build_process_group(req.solution_node_key, simulation_node_key, req.group_type, model_env)
        return BaseResponse(success=True, message=str(group_id))
    except Exception as e:
        return BaseResponse(success=False, message=str(e))

# 4. 启动模拟
@router.post('/start_simulation', response_model=BaseResponse)
def start_simulation(req: StartSimulationRequest):
    simulation_node_key = f'root.simulations.{req.simulation_name}'
    ok = simulation_process_manager.start_simulation(
        req.solution_node_key, simulation_node_key
    )
    return BaseResponse(success=ok, message="Simulation started" if ok else "Failed to start simulation")

# 5. 结束模拟
@router.post('/stop_simulation', response_model=BaseResponse)
def stop_simulation(req: StopSimulationRequest):
    print("Stopping simulation:", req.simulation_node_key)
    ok = simulation_process_manager.stop_simulation(
        req.solution_node_key, req.simulation_node_key
    )
    return BaseResponse(success=ok, message="Simulation stopped" if ok else "Failed to stop simulation")

# 6. 暂停模拟
@router.post('/pause_simulation', response_model=BaseResponse)
def pause_simulation(req: PauseSimulationRequest):
    ok = simulation_process_manager.pause_simulation(
        req.solution_node_key, req.simulation_node_key, req.step
    )
    return BaseResponse(success=ok, message="Simulation paused" if ok else "Failed to pause simulation")

# 7. 恢复模拟
@router.post('/resume_simulation', response_model=BaseResponse)
def resume_simulation(req: ResumeSimulationRequest):
    ok = simulation_process_manager.resume_simulation(
        req.solution_name, req.simulation_name, req.simulation_address
    )
    return {"result": "resumed" if ok else "not running"}

# 8. 获取已完成步骤列表
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

# 9. 获取指定步骤的结果数据
@router.post('/get_step_result', response_model=GetStepResultResponse)
def get_step_result(req: GetStepResultRequest):
    try:
        with BT.instance.connect(req.simulation_node_key, ISimulation, duration=CRMDuration.Forever, reuse=ReuseAction.REPLACE) as simulation:
            result = simulation.get_step_result(req.step)
            if result is not None:
                return GetStepResultResponse(success=True, message="Step result retrieved successfully", result=result)
            else:
                return GetStepResultResponse(success=False, message=f"Step {req.step} is not ready or already retrieved", result={})
    except Exception as e:
        return GetStepResultResponse(success=False, message=str(e), result={})

# 10. 检查步骤是否就绪
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

# 11. 获取模拟状态
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

# 12. 添加人类干预行为
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