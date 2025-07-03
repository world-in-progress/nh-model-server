from fastapi import APIRouter, Body
from src.nh_model_server.schemas.model import Signal, SignalType
from src.nh_model_server.core.simulation import simulation_process_manager
import c_two as cc
from icrms.isolution import ISolution
import os
from src.nh_model_server.core.config import settings

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
        ok = simulation_process_manager.start(solution_name, simulation_name, solution_data, resource_path, simulation_address, body.signal_value.step)

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