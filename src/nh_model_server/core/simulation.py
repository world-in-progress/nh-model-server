import multiprocessing
import threading
import os
from pathlib import Path
from src.nh_model_server.core.monitor import ResultMonitor
import json
from src.nh_model_server.core.config import settings

from model.coupled_0703.Flood_new import run_flood
from model.coupled_0703.pipe_NH import run_pipe_simulation

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

    def start(self, solution_name, simulation_name, solution_data, resource_path, simulation_address=None, step=None):
        key = self._get_key(solution_name, simulation_name)
        with self.lock:
            if key in self.processes:
                procs = self.processes[key]
                if all(proc.poll() is None for proc in procs.values()):
                    return False  # 该任务已在运行

            inp_data = solution_data.get('inp_data')
            inp_path = Path(resource_path)/f"{simulation_name}.inp"
            with open(inp_path, 'w', encoding='utf-8') as f:
                f.write(inp_data)
            solution_data['inp_path'] = inp_path

            # 创建共享内存
            manager = multiprocessing.Manager()
            shared = {
                '1d_data': manager.dict(),
                '2d_data': manager.dict(),
                '1d_ready': manager.Event(),
                '2d_ready': manager.Event(),
                'lock': manager.Lock(),
            }
            env = {}
            env['manager'] = manager
            env['shared'] = shared
            self.envs[key] = env
            
            # 启动两个进程
            flood_proc = multiprocessing.Process(
                target=run_flood,
                args=(shared, solution_data, resource_path, step, 0)
            )
            pipe_proc = multiprocessing.Process(
                target=run_pipe_simulation,
                args=(shared, inp_path, resource_path, step)
            )
            flood_proc.start()
            pipe_proc.start()
            # flood_proc.join()
            # pipe_proc.join()

            # monitor进程
            monitor = ResultMonitor(resource_path, simulation_address, solution_name, simulation_name)
            monitor_proc = multiprocessing.Process(target=monitor.run)
            monitor_proc.start()
            # monitor_proc.join()

            procs = {}
            procs['flood'] = flood_proc
            procs['pipe'] = pipe_proc
            procs['monitor'] = monitor_proc
            self.processes[key] = procs
            return True

    def stop(self, solution_name, simulation_name):
        key = self._get_key(solution_name, simulation_name)
        with self.lock:
            # 停止模拟进程组
            procs = self.processes.get(key)
            if procs:
                for proc in procs.values():
                    if proc.is_alive():
                        proc.terminate()
                        proc.join()
                del self.processes[key]
            env = self.envs.get(key)
            if env:
                env['manager'].shutdown()
                del self.envs[key]
            return True

    def stop_all(self):
        """停止所有进程和监控器"""
        with self.lock:
            # 停止所有进程组
            for key, procs in list(self.processes.items()):
                for proc in procs.values():
                    if proc.is_alive():
                        proc.terminate()
                        proc.join()
            self.processes.clear()
            for env in self.envs.values():
                env['manager'].shutdown()
            self.envs.clear()

    # 可以扩展 rollback, pause, resume 等方法，参数同理加上 key

    def load_process_group_configs(self, path=None):
        if path is None:
            path = os.path.join(settings.DB_PATH, "process_group.json")
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def get_process_group_config(self, group_type):
        for group in self.configs:
            if group["group_type"] == group_type:
                return group
        return None

    def build_process_group(self, solution_name, simulation_name, group_type, env):
        config = self.get_process_group_config(group_type)
        if not config:
            raise ValueError(f"Unknown process group type: {group_type}")
        resource_path = settings.RESOURCE_PATH + "/" + solution_name + "/" + simulation_name
        
        # 只保存进程配置，不创建真实的进程对象
        process_configs = []
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
                    proc_params[pname] = resource_path
                elif pname == "step":
                    proc_params[pname] = 0
                elif pname == "flag":
                    proc_params[pname] = 1
                elif pname in env:
                    origin_path = env[pname]
                    file_name = os.path.basename(origin_path)
                    proc_params[pname] = file_name
                else:
                    raise ValueError(f"Missing parameter: {pname} for process {proc_name}")
            
            # 保存进程配置
            process_config = {
                "name": proc_name,
                "script": proc_cfg["script"],
                "entrypoint": proc_cfg["entrypoint"],
                "params": proc_params
            }
            process_configs.append(process_config)
        
        group_id = self._get_key(solution_name, simulation_name)
        
        self.process_group_info[group_id] = {
            "group_type": group_type,
            "process_configs": process_configs,
            "resource_path": resource_path,
            "shared_config": config.get("shared", []),
            "monitor_config": config.get("monitor_config", {})
        }
        
        print(f"已构建进程组配置，包含 {len(process_configs)} 个进程配置")
        return group_id

    def start_simulation(self, solution_name, simulation_name, simulation_address):
        key = self._get_key(solution_name, simulation_name)
        with self.lock:
            # 检查是否已在运行
            if key in self.processes:
                procs = self.processes[key]
                if all(proc.is_alive() for proc in procs.values()):
                    return False  # 该任务已在运行
            
            if key not in self.process_group_info:
                raise RuntimeError(f"Process group config {key} not found.")
            
            config_info = self.process_group_info[key]
            
            # 创建资源路径
            resource_path = config_info["resource_path"]
            os.makedirs(resource_path, exist_ok=True)
            print("资源路径创建成功！")

            # 获取monitor配置
            monitor_config = config_info["monitor_config"]
            file_types = monitor_config.get("file_types", None)
            file_suffix = monitor_config.get("file_suffix", None)
            
            # 创建跨进程信号
            manager = multiprocessing.Manager()
            stop_event = manager.Event()
            pause_event = manager.Event()
            resume_event = manager.Event()
            update_config_event = manager.Event()
            
            # 创建monitor对象
            monitor = ResultMonitor(
                resource_path, 
                simulation_address,
                solution_name, 
                simulation_name, 
                file_types=file_types,
                file_suffix=file_suffix,
                process_group_config={
                    "group_type": config_info["group_type"],
                    "process_configs": config_info["process_configs"],
                    "resource_path": resource_path,
                    "shared_config": config_info["shared_config"]
                },
                stop_event=stop_event,
                pause_event=pause_event,
                resume_event=resume_event,
                update_config_event=update_config_event
            )
            
            # 创建并启动monitor进程
            monitor_proc = multiprocessing.Process(target=monitor.run)
            monitor_proc.start()
            
            # 保存monitor对象和进程
            monitor_info = {}
            monitor_info["monitor_obj"] = monitor
            monitor_info["monitor_proc"] = monitor_proc
            monitor_info["manager"] = manager
            monitor_info["stop_event"] = stop_event  # 保存停止信号
            monitor_info["pause_event"] = pause_event  # 保存暂停信号
            monitor_info["resume_event"] = resume_event  # 保存继续信号
            monitor_info["update_config_event"] = update_config_event  # 保存配置更新信号
            self.monitors[key] = monitor_info
            
            # 只记录monitor进程
            self.processes[key] = {"monitor": monitor_proc}
            return True
        
    def stop_simulation(self, solution_name, simulation_name):
        key = self._get_key(solution_name, simulation_name)
        with self.lock:
            # 通过Event信号停止monitor进程
            if key in self.monitors:
                monitor_info = self.monitors[key]
                monitor_proc = monitor_info["monitor_proc"]
                stop_event = monitor_info["stop_event"]
                
                print(f"发送停止信号给monitor: {key}")
                stop_event.set()  # 设置停止信号
                
                # 等待进程退出
                if monitor_proc and monitor_proc.is_alive():
                    monitor_proc.join(timeout=10)  # 等待最多10秒
                    if monitor_proc.is_alive():
                        print(f"Monitor进程未能在10秒内退出，强制终止...")
                        monitor_proc.terminate()
                        monitor_proc.join(timeout=5)
                        if monitor_proc.is_alive():
                            print(f"强制杀死Monitor进程")
                            monitor_proc.kill()
                            monitor_proc.join()
                
                # 清理进程记录
                if key in self.processes:
                    del self.processes[key]
                
                # 重置monitor进程
                monitor_info["monitor_proc"] = None
                monitor_info["manager"] = None
                monitor_info["stop_event"] = None
                
            return True
        
    def pause_simulation(self, solution_name, simulation_name):
        key = self._get_key(solution_name, simulation_name)
        with self.lock:
            # 通过Event信号暂停monitor中的子进程
            if key in self.monitors:
                monitor_info = self.monitors[key]
                pause_event = monitor_info.get("pause_event")
                
                if pause_event:
                    print(f"发送暂停信号给monitor: {key}")
                    pause_event.set()  # 设置暂停信号
                    return True
                else:
                    print(f"Monitor {key} 没有暂停事件信号")
                    return False
            else:
                print(f"Monitor {key} 不存在")
                return False

    def resume_simulation(self, solution_name, simulation_name, simulation_address=None, updated_config=None):
        key = self._get_key(solution_name, simulation_name)
        with self.lock:
            # 通过Event信号恢复monitor中的子进程
            if key in self.monitors:
                monitor_info = self.monitors[key]
                resume_event = monitor_info.get("resume_event")
                update_config_event = monitor_info.get("update_config_event")
                
                if resume_event:
                    # 如果有配置更新，先更新配置
                    if updated_config:
                        print(f"更新进程组配置: {key}")
                        # 更新本地配置
                        if key in self.process_group_info:
                            self.process_group_info[key].update(updated_config)
                        
                        # 更新monitor中的配置
                        monitor_obj = monitor_info.get("monitor_obj")
                        if monitor_obj and hasattr(monitor_obj, 'update_config'):
                            monitor_obj.update_config(updated_config.get("process_group_config"))
                        
                        # 设置配置更新信号
                        if update_config_event:
                            update_config_event.set()
                    
                    print(f"发送继续信号给monitor: {key}")
                    resume_event.set()  # 设置继续信号
                    return True
                else:
                    print(f"Monitor {key} 没有继续事件信号")
                    return False
            else:
                print(f"Monitor {key} 不存在")
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
            
            print(f"进程组 {key} 的配置路径已更新")
            return True
            
        except Exception as e:
            print(f"更新进程组配置路径时出错: {e}")
            return False
    
    def get_simulation_status(self, solution_name, simulation_name):
        """获取仿真状态"""
        key = self._get_key(solution_name, simulation_name)
        
        status = {
            "running": False,
            "paused": False,
            "monitor_alive": False,
            "child_processes": {}
        }
        
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