import c_two as cc
from enum import Enum
from pydantic import BaseModel
from typing import Union, Any

class CreateSimulationBody(BaseModel):
    name: str
    solution_name: str
    
class ActionType(str, Enum):
    ADD_FENCE = "add_fence"
    TRANSFER_WATER = "transfer_water"
    ADD_GATE= "add_gate"    
    
class FenceParams(BaseModel):
    elevation_delta: float | None = None
    landuse_type: int
    feature: dict[str, Any]

class TransferWaterParams(BaseModel):
    from_grid: int
    to_grid: int
    q: float  # 通量
    
class GateParams(BaseModel):
    ud_stream: int
    gate_height: int
    feature: dict[str, Any]

class HumanAction(BaseModel):
    action_type: ActionType
    params: Union[TransferWaterParams, FenceParams, GateParams]

class GridResult(BaseModel):
    grid_id: int
    water_level: float
    u: float
    v: float
    depth: float

@cc.icrm
class ISimulation:

    def run(self) -> bool:
        """
        启动模拟
        :return: 启动是否成功
        """
        ...

    def stop(self) -> bool:
        """
        停止模拟
        :return: 停止是否成功
        """
        ...

    # def pause(self) -> bool:
    #     """
    #     暂停模拟
    #     :return: 暂停是否成功
    #     """
    #     ...

    # def resume(self) -> bool:
    #     """
    #     恢复模拟
    #     :return: 恢复是否成功
    #     """
    #     ...


    # def get_human_actions(self, step: int) -> list[HumanAction]:
    #     """
    #     获取人类行为
    #     :param step: 步骤
    #     :return: HumanAction对象列表
    #     """
    #     ...

    # # ------------------------------------------------------------
    # # Front to Resource Server
    # def add_human_action(self, step: int, action: HumanAction) -> dict[str, bool | str]:
    #     """
    #     添加人类行为
    #     :param action: HumanAction对象
    #     """
    #     ...

    # # ------------------------------------------------------------
    # # Result Polling Interface
    # def get_completed_steps(self) -> list[int]:
    #     """
    #     获取已完成但未被拉取的步骤列表
    #     :return: 步骤号列表
    #     """
    #     ...

    def get_step_result(self, step: int) -> dict[str, Any] | None:
        """
        获取指定步骤的结果数据，获取后将该步骤标记为已拉取
        :param step: 步骤号
        :return: 结果数据字典，如果步骤未完成或已被拉取则返回None
        """
        ...

    # def check_step_ready(self, step: int) -> bool:
    #     """
    #     检查指定步骤是否已完成并可以拉取结果
    #     :param step: 步骤号
    #     :return: 是否就绪
    #     """
    #     ...

    # def get_simulation_status(self) -> dict[str, Any]:
    #     """
    #     获取模拟状态，包含已完成步骤信息
    #     :return: 状态字典
    #     """
    #     ...