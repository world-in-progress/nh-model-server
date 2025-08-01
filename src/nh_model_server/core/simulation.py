import os
import json
import threading
from icrms.isimulation import ISimulation
from icrms.itreeger import ReuseAction, CRMDuration
from src.nh_model_server.core.config import settings
from src.nh_model_server.core.bootstrapping_treeger import BT


class SimulationProcessManager:
    def __init__(self):
        self.processes = {}  # key: (solution_name, simulation_name), value: process
        self.lock = threading.Lock()
        self.envs = {}
        self.configs = self.load_process_group_configs()
        self.monitors = {}  # 保存monitor对象和monitor进程: {group_id: {"monitor_obj": obj, "monitor_proc": proc}}
        self.process_group_info = {}  # 保存进程组构建信息

    def _get_key(self, solution_name, simulation_name):
        return (solution_name, simulation_name)

    def load_process_group_configs(self, path=None):
        if path is None:
            path = os.path.join(settings.PERSISTENCE_PATH, "process_group.json")
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def get_process_group_config(self, group_type):
        for group in self.configs:
            if group["group_type"] == group_type:
                return group
        return None

    def build_process_group(self, solution_node_key, simulation_node_key, group_type, model_env):
        config = self.get_process_group_config(group_type)
        if not config:
            raise ValueError(f"Unknown process group type: {group_type}")
        simulation_resource_path = settings.SIMULATION_DIR + "/" + simulation_node_key
        os.makedirs(simulation_resource_path, exist_ok=True)
        print("资源路径创建成功！")

        # 只保存进程配置，不创建真实的进程对象
        process_config_list = []
        # proc_cfg是每一个进程的配置
        for proc_cfg in config["processes"]:
            proc_name = proc_cfg["name"]
            proc_params = {}
            # param_defs是当前遍历的进程的参数配置
            param_defs = proc_cfg["parameters"]
            for param in param_defs:
                pname, ptype = param["name"], param["type"]
                if pname == "shared":
                    proc_params[pname] = "shared"  # 标记，在monitor中会替换为真实的shared对象
                elif pname == "resource_path":
                    proc_params[pname] = simulation_resource_path
                elif pname == "step":
                    proc_params[pname] = 0
                elif pname == "flag":
                    proc_params[pname] = 1
                elif pname in model_env:
                    proc_params[pname] = model_env[pname]
            
            # 保存进程配置
            process_config = {
                "name": proc_name,
                "script": proc_cfg["script"],
                "entrypoint": proc_cfg["entrypoint"],
                "params": proc_params
            }
            process_config_list.append(process_config)
        
        group_id = self._get_key(solution_node_key, simulation_node_key)
        
        self.process_group_info[group_id] = {
            "group_type": group_type,
            "process_configs": process_config_list,
            "shared_config": config.get("shared", []),
            "monitor_config": config.get("monitor_config", {}),
            "helper": config.get("helper", {}),
            "model_env": model_env
        }
        
        print(f"已构建进程组配置，包含 {len(process_config_list)} 个进程配置")
        return group_id

    def start_simulation(self, solution_node_key, simulation_node_key):
        key = self._get_key(solution_node_key, simulation_node_key)
        with self.lock:
            # 检查是否已在运行
            if key in self.processes:
                procs = self.processes[key]
                # 检查模拟线程是否存在且存活
                simulation_thread = procs.get('simulation_thread')
                if simulation_thread and simulation_thread.is_alive():
                    return False  # 该任务已在运行
                
                # 检查其他进程是否存在且存活
                other_procs = {k: v for k, v in procs.items() if k != 'simulation_thread'}
                if other_procs and all(hasattr(proc, 'is_alive') and proc.is_alive() for proc in other_procs.values()):
                    return False  # 该任务已在运行
            
            if key not in self.process_group_info:
                raise RuntimeError(f"Process group config {key} not found.")
            
            config_info = self.process_group_info[key]
            
            params = {
                "solution_node_key": solution_node_key,
                "simulation_node_key": simulation_node_key,
                "process_group_config": config_info,
            }
            BT.instance.mount_node("simulation", simulation_node_key, params)

            try:
                with BT.instance.connect(simulation_node_key, ISimulation, duration=CRMDuration.Forever, reuse=ReuseAction.REPLACE) as simulation:
                    success = simulation.run()
                    print(f"模拟 {simulation_node_key} 启动状态: {success}")
                    if success:
                        print(f"模拟 {simulation_node_key} 启动成功")
                        # 保存节点信息到processes字典中，以便状态查询和停止操作
                        if key not in self.processes:
                            self.processes[key] = {}
                        self.processes[key]['node_key'] = simulation_node_key
                        self.processes[key]['running'] = True
                        return True
                    else:
                        print(f"模拟 {simulation_node_key} 启动失败")
                        return False
            except Exception as e:
                print(f"模拟执行过程中出现错误: {e}")
                return False
        
    def stop_simulation(self, solution_node_key, simulation_node_key):
        key = self._get_key(solution_node_key, simulation_node_key)
        with self.lock:
            try:
                if key in self.processes and 'node_key' in self.processes[key]:
                    node_key = self.processes[key]['node_key']
                    with BT.instance.connect(node_key, ISimulation) as simulation:
                        print(f"正在停止模拟: {simulation_node_key}")
                        success = simulation.stop()
                        if success:
                            # 更新状态
                            self.processes[key]['running'] = False
                            print(f"模拟 {simulation_node_key} 停止成功")
                            return True
                        else:
                            print(f"模拟 {simulation_node_key} 停止失败")
                            return False
                else:
                    print(f"模拟 {simulation_node_key} 未找到或未运行")
                    return False
            except Exception as e:
                print(f"停止模拟时出现错误: {e}")
                return False
        
    def pause_simulation(self, solution_node_key, simulation_node_key, step):
        key = self._get_key(solution_node_key, simulation_node_key)
        with self.lock:
            try:
                if key in self.processes and 'node_key' in self.processes[key]:
                    node_key = self.processes[key]['node_key']
                    with BT.instance.connect(node_key, ISimulation, duration=CRMDuration.Forever, reuse=ReuseAction.KEEP) as simulation:
                        print(f"正在暂停模拟: {simulation_node_key}")
                        success = simulation.pause()
                        if success:
                            # 更新状态
                            self.processes[key]['paused'] = True
                            print(f"模拟 {simulation_node_key} 暂停成功")
                            return True
                        else:
                            print(f"模拟 {simulation_node_key} 暂停失败")
                            return False
                else:
                    print(f"模拟 {simulation_node_key} 未找到或未运行")
                    return False
            except Exception as e:
                print(f"暂停模拟时出现错误: {e}")
                return False

    def resume_simulation(self, solution_name, simulation_name, simulation_address=None, updated_config=None):
        key = self._get_key(solution_name, simulation_name)
        with self.lock:
            try:
                if key in self.processes and 'node_key' in self.processes[key]:
                    node_key = self.processes[key]['node_key']
                    with BT.instance.connect(node_key, ISimulation) as simulation:
                        print(f"正在恢复模拟: {simulation_name}")
                        success = simulation.resume()
                        if success:
                            # 更新状态
                            self.processes[key]['paused'] = False
                            print(f"模拟 {simulation_name} 恢复成功")
                            return True
                        else:
                            print(f"模拟 {simulation_name} 恢复失败")
                            return False
                else:
                    print(f"模拟 {simulation_name} 未找到或未运行")
                    return False
            except Exception as e:
                print(f"恢复模拟时出现错误: {e}")
                return False

    def update_process_config_paths(self, solution_name, simulation_name, new_env_paths):
        """更新进程组配置中的文件路径"""
        key = self._get_key(solution_name, simulation_name)
        
        if key not in self.process_group_info:
            print(f"Process group config {key} not found.")
            return False
            
        try:
            config_info = self.process_group_info[key]
            process_configs = config_info["process_configs"]
            parser_config = config_info.get("parser_config", {})
            data_parsers = parser_config.get("data_parsers", {})
            
            # 更新文件路径映射
            file_paths = config_info.get("file_paths", {})
            for param_name, new_path in new_env_paths.items():
                # 更新文件路径映射
                for data_type in data_parsers.keys():
                    if data_type in param_name.lower() or param_name.lower().endswith(data_type):
                        file_paths[data_type] = new_path
                        print(f"更新数据类型 {data_type} 的文件路径: {new_path}")
                        break
            
            # 更新每个进程配置中的路径参数
            for proc_config in process_configs:
                proc_params = proc_config["params"]
                
                # 更新路径相关的参数
                for param_name, new_path in new_env_paths.items():
                    if param_name in proc_params:
                        if param_name == "resource_path":
                            # 直接使用新的资源路径
                            proc_params[param_name] = new_path
                        else:
                            # 对于其他文件路径，提取文件名
                            file_name = os.path.basename(new_path)
                            proc_params[param_name] = file_name
                            print(f"更新进程 {proc_config['name']} 的参数 {param_name}: {file_name}")
            
            # 保存更新后的文件路径映射
            config_info["file_paths"] = file_paths
            
            # 如果monitor正在运行，也更新monitor中的文件路径
            if key in self.monitors:
                monitor_info = self.monitors[key]
                monitor_obj = monitor_info.get("monitor_obj")
                if monitor_obj and hasattr(monitor_obj, 'update_file_paths'):
                    monitor_obj.update_file_paths(file_paths)
            
            print(f"进程组 {key} 的配置路径已更新")
            return True
            
        except Exception as e:
            print(f"更新进程组配置路径时出错: {e}")
            return False
    
    def get_simulation_status(self, solution_name, simulation_name):
        """获取模拟状态"""
        key = self._get_key(solution_name, simulation_name)
        
        status = {
            "running": False,
            "paused": False,
            "monitor_alive": False,
            "simulation_thread_alive": False,
            "child_processes": {}
        }
        
        # 检查模拟状态
        if key in self.processes:
            proc_info = self.processes[key]
            
            # 从进程信息中获取运行状态
            if proc_info.get('running', False):
                status["running"] = True
                
                # 尝试连接到模拟实例获取更详细的状态
                if 'node_key' in proc_info:
                    try:
                        node_key = proc_info['node_key']
                        with BT.instance.connect(node_key, ISimulation) as simulation:
                            # 使用新的状态查询方法
                            sim_status = simulation.get_status()
                            status.update({
                                "running": sim_status.get("running", False),
                                "paused": sim_status.get("paused", False),
                                "simulation_thread_alive": sim_status.get("monitor_thread_alive", False),
                                "monitor_alive": sim_status.get("monitor_thread_alive", False),
                                "current_step": sim_status.get("current_step", 0),
                                "child_processes_count": sim_status.get("child_processes_count", 0)
                            })
                    except Exception as e:
                        print(f"获取模拟详细状态时出错: {e}")
                        # 如果无法获取详细状态，至少标记为运行中
                        status["running"] = proc_info.get('running', False)
        
        # 检查monitor状态（保持兼容性）
        if key in self.monitors:
            monitor_info = self.monitors[key]
            monitor_proc = monitor_info.get("monitor_proc")
            monitor_obj = monitor_info.get("monitor_obj")
            
            if monitor_proc and monitor_proc.is_alive():
                status["monitor_alive"] = True
                status["running"] = True
                
                # 如果能访问monitor对象，获取更详细的状态
                if monitor_obj:
                    status["paused"] = getattr(monitor_obj, 'paused', False)
                    status["current_step"] = getattr(monitor_obj, 'current_step', 0)
        
        return status

simulation_process_manager = SimulationProcessManager()