import c_two as cc
from enum import Enum
from pydantic import BaseModel
from dataclasses import dataclass

class ScenarioNodeType(Enum):
    Unknown = 0
    Resource = 1
    Conception = 2
    Aggregation = 3

class CRMEntry(BaseModel):
    name: str
    icrm: str
    crm_launcher: str

class ScenarioNode(BaseModel):
    name: str
    crm: str | None = None
    semantic_path: str = ''
    parent: 'ScenarioNode' = None
    children: list['ScenarioNode'] = []
    node_type: ScenarioNodeType = ScenarioNodeType.Unknown

class ScenarioNodeDescription(BaseModel):
    semanticPath: str
    children: list[str] = []

class TreeConfiguration(BaseModel):
    scene_path: str
    max_ports: int = 0
    port_range: tuple[int, int] = (0, 0)

class TreeMeta(BaseModel):
    scenario: ScenarioNode
    crm_entries: list[CRMEntry]
    configuration: TreeConfiguration

class ReuseAction(Enum):
    KEEP = 0
    FORK = 1
    REPLACE = 2

@dataclass
class SceneNodeInfo:
    node_key: str
    scenario_node_name: str
    parent_key: str | None = None
    server_address: str | None = None

class SceneNodeMeta(BaseModel):
    node_key: str
    scenario_path: str
    children: list['SceneNodeMeta'] | None = None

class CRMDuration(Enum):
    Once = '0'
    Much_Short = '5'
    Very_Short = '10'
    Short = '30'
    Medium = '60'
    Long = '120'
    Very_Long = '300'
    Much_Long = '600'
    Forever = '-1'
    
@cc.icrm
class ITreeger:
    def mount_node(self, scenario_node_name: str, node_key: str, launch_params: dict | None = None) -> None:
        ...
    
    def unmount_node(self, node_key: str) -> bool:
        ...
        
    def activate_node(self, node_key: str, reusibility: ReuseAction = ReuseAction.REPLACE, duration: CRMDuration = CRMDuration.Medium) -> str:
        ...
        
    def deactivate_node(self, node_key: str) -> bool:
        ...
    
    def get_node_info(self, node_key: str) -> SceneNodeInfo | None:
        ...
    
    def get_process_pool_status(self) -> dict:
        ...
    
    def get_scene_node_info(self, node_key: str, child_start_index: int = 0, child_end_index: int | None = None) -> SceneNodeMeta | None:
        ...
    
    def get_scenario_description(self) -> list[ScenarioNodeDescription]:
        ...