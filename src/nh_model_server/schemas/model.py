from pydantic import BaseModel
from enum import Enum

class SolutionCheckRequest(BaseModel):
    solution: dict

class CloneEnvRequest(BaseModel):
    solution_name: str
    solution_address: str

class CloneActionRequest(BaseModel):
    solution_name: str
    solution_address: str

class BuildProcessGroupRequest(BaseModel):
    solution_name: str
    simulation_name: str
    group_type: str
    solution_address: str

class StartSimulationRequest(BaseModel):
    solution_name: str
    simulation_name: str
    simulation_address: str

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