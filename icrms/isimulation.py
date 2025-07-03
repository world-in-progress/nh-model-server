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

class TransferWaterParams(BaseModel):
    from_grid: int
    to_grid: int
    q: float  # 通量

class AddGateParams(BaseModel):
    gate_num: int
    grid_id_list: list[int]
    ud_stream: int
    gate_height: int

class HumanAction(BaseModel):
    action_type: ActionType
    params: Union[AddFenceParams, TransferWaterParams, AddGateParams]

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

    def send_result(self, step: int, result: list[GridResult], highlight_grids: list[int]) -> dict[str, bool | str]:
        """
        发送结果
        :param step: 步骤
        :param result: 结果
        :param highlight_grids: 特殊网格
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