import subprocess
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
        self.process_group_configs = self.load_process_group_configs()
        self.process_groups = {}  # 保存已构建但未执行的进程组

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
        for group in self.process_group_configs:
            if group["group_type"] == group_type:
                return group
        return None

    def build_shared_structure(self, shared_config):
        manager = multiprocessing.Manager()
        shared = {}
        for item in shared_config:
            name, typ = item["name"], item["type"]
            if typ == "dict":
                shared[name] = manager.dict()
            elif typ == "Event":
                shared[name] = manager.Event()
            elif typ == "Lock":
                shared[name] = manager.Lock()
            else:
                raise ValueError(f"Unsupported shared type: {typ}")
        return shared, manager

    def build_process_group(self, solution_name, simulation_name, group_type, env):
        import importlib
        config = self.get_process_group_config(group_type)
        if not config:
            raise ValueError(f"Unknown process group type: {group_type}")
        shared, manager = self.build_shared_structure(config.get("shared", []))
        resource_path = settings.RESOURCE_PATH + "/" + solution_name + "/" + simulation_name
        processes = {}
        # proc_cfg是每一个进程的配置
        for proc_cfg in config["processes"]:
            proc_name = proc_cfg["name"]
            proc_params = {}
            # param_defs是当前遍历的进程的参数配置
            param_defs = proc_cfg["parameters"]
            for param in param_defs:
                pname, ptype = param["name"], param["type"]
                if pname == "shared":
                    proc_params[pname] = shared
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
            # 动态import脚本，获取entrypoint
            script_path = proc_cfg["script"]
            entrypoint = proc_cfg["entrypoint"]
            # 支持相对路径
            if not os.path.isabs(script_path):
                module_path = settings.MODEL_PATH + "." + script_path
            module = importlib.import_module(module_path)
            entrypoint_func = getattr(module, entrypoint)
            proc = multiprocessing.Process(target=entrypoint_func, kwargs=proc_params)
            processes[proc_name] = proc
        group_id = f"{solution_name}_{simulation_name}"
        print(group_id)
        self.process_groups[group_id] = {
            "group_type": group_type,
            "shared": shared,
            "manager": manager,
            "processes": processes,
            "resource_path": resource_path
        }
        print(self.process_groups[group_id]["processes"])
        return group_id

    # 启动模拟时才会将solution运行为simulation从而得到simulation_address
    def start_simulation(self, solution_name, simulation_name, simulation_address):
        key = self._get_key(solution_name, simulation_name)
        group_id = f"{solution_name}_{simulation_name}"
        with self.lock:
            # 检查是否已在运行
            if key in self.processes:
                procs = self.processes[key]
                if all(proc.is_alive() for proc in procs.values()):
                    return False  # 该任务已在运行
            # 检查进程组是否已构建
            if group_id not in self.process_groups:
                raise RuntimeError(f"Process group {group_id} not built. Please build it first.")
            group = self.process_groups[group_id]
            
            # 创建资源路径
            resource_path = group["resource_path"]
            os.makedirs(resource_path, exist_ok=True)

            # 创建并启动monitor进程
            config = self.get_process_group_config(group["group_type"])
            monitor_config = config.get("monitor_config", {})
            file_types = monitor_config.get("file_types", None)
            file_suffix = monitor_config.get("file_suffix", None)
            monitor = ResultMonitor(
                resource_path, 
                simulation_address, 
                solution_name, 
                simulation_name, 
                file_types=file_types, 
                file_suffix=file_suffix
            )
            monitor_proc = multiprocessing.Process(target=monitor.run)
            
            # 启动进程组内所有进程
            procs = group["processes"]
            for proc in procs.values():
                if not proc.is_alive():
                    proc.start()
            
            # 启动monitor进程
            monitor_proc.start()
            
            # 记录所有进程（包括monitor）
            all_procs = dict(procs)
            all_procs["monitor"] = monitor_proc
            self.processes[key] = all_procs
            return True
        
    def stop_simulation(self, solution_name, simulation_name):
        key = self._get_key(solution_name, simulation_name)
        with self.lock:
            if key in self.processes:
                procs = self.processes[key]
                for proc in procs.values():
                    if proc.is_alive():
                        proc.terminate()
                        proc.join()
                del self.processes[key]
            # TODO: 删除资源路径
            return True
        
    def pause_simulation(self, solution_name, simulation_name):
        key = self._get_key(solution_name, simulation_name)
        with self.lock:
            if key in self.processes:
                procs = self.processes[key]
                for proc in procs.values():
                    if proc.is_alive():
                        proc.terminate()
                        proc.join()
            return True

    def resume_simulation(self, solution_name, simulation_name, simulation_address):
        key = self._get_key(solution_name, simulation_name)
        with self.lock:
            # TODO: 获取human_action
            if key in self.processes:
                procs = self.processes[key]
                for proc in procs.values():
                    if not proc.is_alive():
                        proc.start()
            return True

simulation_process_manager = SimulationProcessManager()