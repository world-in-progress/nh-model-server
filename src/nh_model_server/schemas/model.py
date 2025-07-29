from pydantic import BaseModel
from enum import Enum
from typing import Any

class SolutionCheckRequest(BaseModel):
    solution: dict

class ClonePackageRequest(BaseModel):
    solution_node_key: str
    solution_address: str

class BuildProcessGroupRequest(BaseModel):
    solution_name: str
    simulation_name: str
    group_type: str
    solution_address: str

class StartSimulationRequest(BaseModel):
    solution_name: str
    simulation_name: str

class StopSimulationRequest(BaseModel):
    solution_name: str
    simulation_name: str

class PauseSimulationRequest(BaseModel):
    solution_name: str
    simulation_name: str

class ResumeSimulationRequest(BaseModel):
    solution_name: str
    simulation_name: str
    simulation_address: str

class GetCompletedStepsRequest(BaseModel):
    solution_name: str
    simulation_name: str
    simulation_address: str

class GetStepResultRequest(BaseModel):
    solution_name: str
    simulation_name: str
    simulation_address: str
    step: int

class CheckStepReadyRequest(BaseModel):
    solution_name: str
    simulation_name: str
    simulation_address: str
    step: int

class GetSimulationStatusRequest(BaseModel):
    solution_name: str
    simulation_name: str
    simulation_address: str

class AddHumanActionRequest(BaseModel):
    solution_name: str
    simulation_name: str
    simulation_address: str
    step: int
    action: dict[str, Any]  # HumanAction的字典形式