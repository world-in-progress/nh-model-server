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

class LanduseType(str, Enum):
    POND = "pond"
    FENCE = "fence"
    DRAIN = "drain"
    DAM = "dam"

class AddFenceParams(BaseModel):
    elevation_delta: float | None = None
    landuse_type: LanduseType | None = None
    feature: dict[str, Any]
    
class FenceParams(BaseModel):
    elevation_delta: float | None = None
    landuse_type: LanduseType | None = None
    grid_id_list: list[int]

class TransferWaterParams(BaseModel):
    from_grid: int
    to_grid: int
    q: float  # 通量

class AddGateParams(BaseModel):
    ud_stream: int
    gate_height: int
    feature: dict[str, Any]
    
class GateParams(BaseModel):
    ud_stream: int
    gate_height: int
    grid_id_list: list[int]

class HumanAction(BaseModel):
    action_type: ActionType
    params: Union[AddFenceParams, TransferWaterParams, AddGateParams, FenceParams, GateParams]

class GridResult(BaseModel):
    grid_id: int
    water_level: float
    u: float
    v: float
    depth: float

@cc.icrm
class ISimulation:

    def get_human_actions(self, step: int) -> list[HumanAction]:
        """
        获取人类行为
        :param step: 步骤
        :return: HumanAction对象列表
        """
        ...

    def send_result(self, step: int, result_data: dict[str, Any], file_types: list[str], file_suffix: dict[str, str]) -> dict[str, bool | str]:
        """
        发送结果
        :param step: 步骤
        :param result_data: 结果数据
        """
        ...

    # ------------------------------------------------------------
    # Front to Resource Server
    def add_human_action(self, step: int, action: HumanAction) -> dict[str, bool | str]:
        """
        添加人类行为
        :param action: HumanAction对象
        """
        ...