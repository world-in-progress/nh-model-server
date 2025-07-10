from pydantic import BaseModel
from enum import Enum

class SignalType(str, Enum):
    START = 'start'
    STOP = 'stop'
    ROLLBACK = 'rollback'
    PAUSE = 'pause'
    RESUME = 'resume'

class SignalValue(BaseModel):
    solution_name: str
    simulation_name: str
    solution_address: str
    simulation_address: str
    step: int | None = None

class Signal(BaseModel):
    signal_type: SignalType
    signal_value: SignalValue

class SolutionCheckRequest(BaseModel):
    solution: dict

class CloneEnvRequest(BaseModel):
    solution_name: str
    solution_address: str

class BuildProcessGroupRequest(BaseModel):
    solution_name: str
    simulation_name: str
    group_type: str
    process_args: dict = {}

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