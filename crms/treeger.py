import os
import sys
import time
import yaml
import json
import sqlite3
import logging
import threading
import subprocess
import c_two as cc
from pathlib import Path
from dotenv import load_dotenv
from typing import TypeVar, Type
from contextlib import contextmanager
from dataclasses import dataclass, field
from icrms.itreeger import ITreeger, CRMEntry, TreeMeta, ReuseAction, ScenarioNode, ScenarioNodeType, SceneNodeInfo, SceneNodeMeta, ScenarioNodeDescription, CRMDuration

T = TypeVar('T')

load_dotenv()
logger = logging.getLogger(__name__)

@dataclass
class ProcessInfo():
    address: str
    start_time: float = 0.0
    scenario_node_name: str = ''
    process: subprocess.Popen | None = None

@dataclass
class SceneNode():
    node_key: str
    scenario_node: ScenarioNode
    launch_params: dict = field(default_factory=dict)
    
    parent: 'SceneNode' = None
    children: list['SceneNode'] = field(default_factory=list)

    def add_parent(self, parent: 'SceneNode'):
        self.parent = parent
        if parent:
            parent.children.append(self)
    
    def add_child(self, child: 'SceneNode'):
        self.children.append(child)
        self.children.sort(key=lambda child: child.node_key.split('.')[-1].lower())  # sort children by their name
        child.parent = self
    
    def add_children(self, children: list['SceneNode']):
        for child in children:
            self.add_child(child)

@cc.iicrm
class Treeger(ITreeger):
    def __init__(self):
        meta_path = os.getenv('SCENARIO_META_PATH', None)
        if not meta_path:
            raise ValueError('SCENARIO_META_PATH environment variable is not set, Treeger cannot be initialized')
        
        self.lock = threading.RLock()
        self.meta_path = Path(meta_path)
        self.process_pool: dict[str, ProcessInfo] = {}
        self.scene_nodes_in_flight: dict[str, set[str]] = {}  # scenario node name -> set of scene node names
        
        with open(meta_path, 'r') as f:
            tree_meta = yaml.safe_load(f)
        self.meta = TreeMeta(**(tree_meta['meta']))
        self.crm_entry_dict: dict[str, CRMEntry] = {
            node.name: node for node in self.meta.crm_entries
        }

        # Iterate through the scenario
        self.root = self.meta.scenario
        self.root.parent = None
        self.root.semantic_path = self.root.name
        self.scenario_node_dict: dict[str, ScenarioNode] = {
            self.root.name: self.root
        }
        scenario_node_stack = [self.root]
        while scenario_node_stack:
            # Get the current scenario node
            scenario_node = scenario_node_stack.pop()
            # Record the scenario node in the dictionary
            self.scenario_node_dict[scenario_node.name] = scenario_node
            # Initialize the CRM in-flight set if not exists
            if scenario_node.name not in self.scene_nodes_in_flight:
                self.scene_nodes_in_flight[scenario_node.name] = set()

            # Parse node type
            if len(scenario_node.children) == 0:
                scenario_node.node_type = ScenarioNodeType.Resource
            elif len(scenario_node.children) == 1:
                scenario_node.node_type = ScenarioNodeType.Conception
            else:
                scenario_node.node_type = ScenarioNodeType.Aggregation
                
            # Update semantic paths for all children
            for child in scenario_node.children:
                child.parent = scenario_node
                child.semantic_path = f'{scenario_node.semantic_path}.{child.name}'
                scenario_node_stack.append(child)
        
        # Initialize scene db
        self.scene_db_path = Path(self.meta.configuration.scene_path)
        self._init_db()
            
    def _init_db(self):
        # Create database directory if it doesn't exist
        self.scene_db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create the database file if it doesn't exist
        with sqlite3.connect(self.scene_db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS scene_nodes (
                    node_key TEXT PRIMARY KEY,
                    scenario_node_name TEXT NOT NULL,
                    launch_params TEXT,
                    parent_key TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (parent_key) REFERENCES scene_nodes (node_key) ON DELETE CASCADE
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_parent_key ON scene_nodes(parent_key)")
            conn.execute("CREATE INDEX IF NOT EXISTS ids_scenario_node_name ON scene_nodes(scenario_node_name)")
            conn.commit()
    
    @contextmanager
    def _connect_db(self):
        """Context manager for database connection."""
        conn = sqlite3.connect(self.scene_db_path)
        conn.row_factory = sqlite3.Row  # enable column access by name
        try:
            yield conn
        finally:
            conn.close()
    
    def _node_exists_in_db(self, node_key: str) -> bool:
        """Check if a node exists in the database"""
        with self._connect_db() as conn:
            cursor = conn.execute("SELECT 1 FROM scene_nodes WHERE node_key = ?", (node_key,))
            return cursor.fetchone() is not None
    
    def _insert_node_to_db(self, node_key: str, scenario_node_name: str, launch_params: dict | None, parent_key: str | None) -> None:
        """Insert a new node into the database"""
        with self._connect_db() as conn:
            conn.execute("""
                INSERT INTO scene_nodes (node_key, scenario_node_name, launch_params, parent_key)
                VALUES (?, ?, ?, ?)
            """, (
                node_key,
                scenario_node_name,
                json.dumps(launch_params) if launch_params else None,
                parent_key if parent_key else None
            ))
            conn.commit()
    
    def _get_child_keys_from_db(self, parent_key: str) -> list[str]:
        """Get all child node keys for a given parent from databse"""
        with self._connect_db() as conn:
            cursor = conn.execute("SELECT node_key FROM scene_nodes WHERE parent_key = ?", (parent_key,))
            return [row['node_key'] for row in cursor.fetchall()]
    
    def _delete_node_from_db(self, node_key: str) -> None:
        """Delete a node from the database"""
        with self._connect_db() as conn:
            conn.execute("DELETE FROM scene_nodes WHERE node_key = ?", (node_key,))
            conn.commit()
    
    def _load_node_from_db(self, node_key: str) -> SceneNode | None:
        """Load a single node from the database"""
        with self._connect_db() as conn:
            cursor = conn.execute("""
                SELECT node_key, scenario_node_name, launch_params, parent_key
                FROM scene_nodes
                WHERE node_key = ?
            """, (node_key,))
            row = cursor.fetchone()
            if row is None:
                return None
            
            scenario_node = self.scenario_node_dict.get(row['scenario_node_name'])
            if scenario_node is None:
                logger.error(f'Scenario node {row["scenario_node_name"]} not found in tree meta')
                return None
            
            launch_params = json.loads(row['launch_params']) if row['launch_params'] else {}
            node = SceneNode(
                node_key=row['node_key'],
                scenario_node=scenario_node,
                launch_params=launch_params
            )
            
            children = self._get_child_keys_from_db(node_key)
            if children:
                with self._connect_db() as conn:
                    cursor = conn.execute("""
                        SELECT node_key, scenario_node_name FROM scene_nodes WHERE parent_key = ?
                    """, (node_key,))
                    
                    for child_row in cursor.fetchall():
                        child_scenario_node = self.scenario_node_dict.get(child_row['scenario_node_name'])
                        if child_scenario_node:
                            child_node = SceneNode(
                                node_key=child_row['node_key'],
                                scenario_node=child_scenario_node,
                                launch_params={}
                            )
                            node.add_child(child_node)
                        else:
                            logger.error(f'Scenario node {child_row["scenario_node_name"]} not found in tree meta')
            
            return node
    
    def _release_crm_process(self, node_key: str):
        if node_key in self.process_pool:
            process_info = self.process_pool[node_key]
            
            # Remove record from process pool and scene node in-flight set
            del self.process_pool[node_key]
            self.scene_nodes_in_flight[process_info.scenario_node_name].remove(node_key)
    
    def _cleanup_finished_processes(self):
        finished_nodes = []
        
        for node_name, node_info in self.process_pool.items():
            process = node_info.process
            if process and process.poll() is not None:
                finished_nodes.append(node_name)
        
        for node_name in finished_nodes:
            self._release_crm_process(node_name)

    def mount_node(self, scenario_node_name: str, node_key: str, launch_params: dict | None = None) -> None:
        with self.lock:
            # Check if node already exists in db
            if (self._node_exists_in_db(node_key)):
                logger.debug(f'Node {node_key} already mounted, skipping')
                return
            
            scenario_node = self.scenario_node_dict.get(scenario_node_name, None)
            if scenario_node is None:
                logger.error(f'Scenario node {scenario_node_name} not found in tree meta')
                raise ValueError(f'Scenario node {scenario_node_name} not found in tree meta')
            
            if not scenario_node.crm and launch_params is not None:
                logger.warning(f'Launch parameters provided for node "{scenario_node_name}" not having a CRM, ignoring launch_params {launch_params}')
                launch_params = {}
            
            # Validate node_key
            parent_key = '.'.join(node_key.split('.')[:-1])
            if parent_key and not self._node_exists_in_db(parent_key):
                raise ValueError(f'Parent node "{parent_key}" not found in scene for node "{node_key}"')

            # Insert into db
            self._insert_node_to_db(node_key, scenario_node_name, launch_params, parent_key if parent_key else None)
            
            logger.info(f'Successfully mounted node "{node_key}" for scenario "{scenario_node_name}"')
    
    def _unmount_node_recursively(self, node_key: str) -> bool:
        if not self._node_exists_in_db(node_key):
            logger.warning(f'Node "{node_key}" not found in scene, cannot unmount')
            return False
        
        # Get all child nodes from database
        child_keys = self._get_child_keys_from_db(node_key)
        
        # Recursively unmount all children
        for child_key in child_keys:
            self._unmount_node_recursively(child_key)
        
        # Stop the node service if it is running
        if node_key in self.process_pool:
            self.deactivate_node(node_key)
        
        # Remove from database
        self._delete_node_from_db(node_key)
            
        logger.info(f'Successfully unmounted node {node_key}')
        return True

    def unmount_node(self, node_key: str) -> bool:
        with self.lock:
            return self._unmount_node_recursively(node_key)
        
        scene_path = ROOT_DIR / self.meta.configuration.scene_path
        with open(scene_path, 'w') as f:
            yaml.dump(scene_data, f, default_flow_style=False)
        logger.info(f'Scene serialized to {scene_path}')

    def terminate(self) -> bool:
        with self.lock:
            try:
                for node_key in list(self.process_pool.keys()):
                    self.deactivate_node(node_key)
                
                logger.info('All nodes stopped successfully')
                
                return True
            except Exception as e:
                logger.error(f'Failed to terminate treeger: {e}')
                return False
    
    def activate_node(self, node_key: str, reusibility: ReuseAction = ReuseAction.REPLACE, duration: CRMDuration = CRMDuration.Medium) -> str:
        with self.lock:
            self._cleanup_finished_processes()
            # Check if the node exists in the db
            if not self._node_exists_in_db(node_key):
                logger.error(f'Node "{node_key}" not found in scene or database')
                raise ValueError(f'Node "{node_key}" not found in scene or database')
            else:
                node = self._load_node_from_db(node_key)
            
            # Check if the node can be launched
            if not node.scenario_node.crm:
                raise ValueError(f'Node {node_key} does not have a CRM and cannot be launched directly')

            # Check if the node is already running
            if node_key in self.process_pool:
                process_info = self.process_pool[node_key]
                return process_info.address
            
            # Handle reusability actions
            flying_sibling_set = self.scene_nodes_in_flight.get(node.scenario_node.name)
            # Get the first available node sharing the same scenario node
            sibling_node_name = next(iter(flying_sibling_set), None)
            if sibling_node_name:
                if reusibility == ReuseAction.KEEP:
                    # Keep the crm process
                    sibling_process_info = self.process_pool.get(sibling_node_name)
                    return sibling_process_info.address

                elif reusibility == ReuseAction.REPLACE:
                    # Replace the sibling node with the new one (stop the sibling process and create below)
                    self.deactivate_node(sibling_node_name)

                elif reusibility == ReuseAction.FORK:
                    # Fork the sibling node, which means creating a new process for the node but keeping the sibling process running
                    pass

            # Try to allocate an address for the node
            try:
                address = f'memory://{node_key.replace("/", "_")}'
            except Exception as e:
                logger.error(f'Failed to allocate address for node {node_key}: {e}')
                raise

            # Try to launch a CRM server related to the node
            try:
                # Platform-specific subprocess arguments
                kwargs = {}
                if sys.platform != 'win32':
                    # Unix-specific: create new process group
                    kwargs['preexec_fn'] = os.setsid
                
                # Assmble the command to launch the CRM server
                params = node.launch_params
                crm_entry: CRMEntry = self.crm_entry_dict.get(node.scenario_node.crm, None)
                if crm_entry is None:
                    raise ValueError(f'CRM template {node.scenario_node.crm} not found in tree meta')
                
                cmd = [
                    sys.executable,
                    crm_entry.crm_launcher,
                    '--server_address', address,
                    '--timeout', str(duration.value),
                ]
                if params:
                    for key, value in params.items():
                        if isinstance(value, dict) or isinstance(value, list):
                            json_str = json.dumps(value, ensure_ascii=False)
                            if sys.platform == 'win32':
                                cmd.extend([f'--{key}', json_str])
                            else:
                                cmd.extend([f'--{key}', f"'{json_str}'"])
                        else:
                            cmd.extend([f'--{key}', str(value)])
                
                process = subprocess.Popen(
                    cmd,
                    **kwargs
                )
                
                # Register the process in the process pool and scene node in-flight set
                self.process_pool[node_key] = ProcessInfo(
                    address=address,
                    process=process,
                    start_time=time.time(),
                    scenario_node_name=node.scenario_node.name
                )
                self.scene_nodes_in_flight[node.scenario_node.name].add(node_key)
                
                # Pin the crm server
                while True:
                    if cc.rpc.Client.ping(address, timeout=1):
                        break
                    if time.time() - self.process_pool[node_key].start_time > 60:
                        raise RuntimeError(f'Timeout waiting for node "{node_key}" to start')
                    
                    time.sleep(0.1)

                logger.info(f'Successfully launched node "{node_key}" at {address}')
                return address

            except Exception as e:
                logger.error(f'Failed to launch node {node_key}: {e}')
                raise

    def deactivate_node(self, node_key: str) -> bool:
        with self.lock:
            if node_key not in self.process_pool:
                logger.warning(f'Node "{node_key}" not found in process pool')
                return False
            
            try:
                process_info = self.process_pool[node_key]
                server_address = process_info.address
                if cc.rpc.Client.shutdown(server_address, timeout=60) is False:
                    raise RuntimeError(f'Failed to shutdown node "{node_key}" at {server_address}')
                else:
                    process = process_info.process
                    if process and process.poll() is None:
                        process.terminate()
                        process.wait()

                # Remove record from process pool and scene node in-flight set
                del self.process_pool[node_key]
                self.scene_nodes_in_flight[process_info.scenario_node_name].remove(node_key)
                
                # Cleanup finished processes
                self._cleanup_finished_processes()
                logger.info(f'Successfully stopped node "{node_key}"')
                return True
            
            except Exception as e:
                logger.error(f'Failed to stop node "{node_key}": {e}')
                return False

    def get_scene_node_info(self, node_key: str, child_start_index: int = 0, child_end_index: int | None = None) -> SceneNodeMeta | None:
        with self.lock:
            # Check if the node exists in the scene
            if self._node_exists_in_db(node_key):
                # Load the node from database if it exists
                scene_node = self._load_node_from_db(node_key)
                if not scene_node:
                    logger.warning(f'Node "{node_key}" not found in scene or database')
                    return None

            # Get the SceneNode instance
            scenario_node_path = scene_node.scenario_node.semantic_path
            
            child_start_index = min(child_start_index, len(scene_node.children))
            child_end_index = len(scene_node.children) if child_end_index is None else min(child_end_index, len(scene_node.children))

            # Get meta of children nodes
            children_meta: list[SceneNodeMeta] = []
            for child in scene_node.children[child_start_index:child_end_index]:
                children_meta.append(SceneNodeMeta(
                    node_key=child.node_key,
                    scenario_path=child.scenario_node.semantic_path,
                    children=None  # do not focus on children meta of children
                ))

            return SceneNodeMeta(
                node_key=scene_node.node_key,
                scenario_path=scenario_node_path,
                children=children_meta if children_meta else None
            )
    
    def get_scenario_description(self) -> list[ScenarioNodeDescription]:
        with self.lock:
            description: list[ScenarioNodeDescription] = []
            
            for scenario_node in self.scenario_node_dict.values():
                description.append(
                    ScenarioNodeDescription(
                        semanticPath=scenario_node.semantic_path,
                        children=[child.name for child in scenario_node.children]
                    )
                )
                
            return description
    
    def get_node_info(self, node_key: str) -> SceneNodeInfo | None:
        with self.lock:
            self._cleanup_finished_processes()
            
            # Check if the node exists in db
            if not self._node_exists_in_db(node_key):
                logger.warning(f'Node "{node_key}" not found in database')
                return None
            
            # Get the SceneNode instance
            scene_node = self._load_node_from_db(node_key)
        
            # Initialize server_address to None by default
            server_address = None
            
            # Get the server address of the node if it is running
            if node_key in self.process_pool:
                process_info = self.process_pool[node_key]
                if process_info.process and process_info.process.poll() is None:
                    # Process is running, return its address
                    server_address = process_info.address
                else:
                    # Remove record from process pool and scene node in-flight set
                    del self.process_pool[node_key]
                    self.scene_nodes_in_flight[process_info.scenario_node_name].remove(node_key)
            
            # Prepare the node info
            node_info = SceneNodeInfo(
                node_key=scene_node.node_key,
                scenario_node_name=scene_node.scenario_node.name,
                parent_key=scene_node.parent.node_key if scene_node.parent else None,
                server_address=server_address
            )
            return node_info

    def get_process_pool_status(self) -> dict:
        with self.lock:
            self._cleanup_finished_processes()
            running_nodes = []
            for node_name, node_info in self.process_pool.items():
                process = node_info.process
                status = 'running' if process and process.poll() is None else 'stopped'
                running_nodes.append({
                    'status': status,
                    'name': node_name,
                    'address': node_info.address,
                    'template': node_info.scenario_node_name,
                    'uptime': time.time() - node_info.start_time
                })
            
            return {
                'nodes': running_nodes,
            }
    
    def trigger(self, node_key: str, proxy_crm_class: Type[T]) -> T | None:
        """Can only be used in the same thread by other CRM"""
        # TODO: make icrm proxy for remote CRM
        # TODO: NOT ROBUST for icrm proxy
        is_crm = proxy_crm_class.direction == '<-'
        
        with self.lock:
            if not self._node_exists_in_db(node_key):
                return None
            else:
                node = self._load_node_from_db(node_key)
                if not node:
                    return None
                if is_crm:
                    return proxy_crm_class(**node.launch_params)
                else:
                    address = self.activate_node(node_key, ReuseAction.KEEP, CRMDuration.Forever)
                    client = cc.rpc.Client(address)
                    proxy = proxy_crm_class()
                    proxy.client = client
                    proxy.close = lambda: client.close()
                    return proxy