import subprocess
import multiprocessing
import threading
from pathlib import Path
import tempfile
import json
from src.nh_model_server.core.monitor import ResultMonitor

from model.coupled_0703.Flood_new import run_flood
from model.coupled_0703.pipe_NH import run_pipe_simulation

class SimulationProcessManager:
    def __init__(self):
        self.processes = {}  # key: (solution_name, simulation_name), value: process
        self.lock = threading.Lock()
        self.managers = {}  # key: (solution_name, simulation_name), value: manager

    def _get_key(self, solution_name, simulation_name):
        return (solution_name, simulation_name)

    def start(self, solution_name, simulation_name, solution_data, resource_path, simulation_address=None, step=None):
        key = self._get_key(solution_name, simulation_name)
        with self.lock:
            if key in self.processes:
                procs = self.processes[key]
                if all(proc.poll() is None for proc in procs.values()):
                    return False  # 该任务已在运行

            # 写入solution_data到临时json文件
            with tempfile.NamedTemporaryFile('w', delete=False, suffix='.json', encoding='utf-8') as tmp_json:
                json.dump(solution_data, tmp_json, ensure_ascii=False)
                tmp_json_path = tmp_json.name

            # 构造命令行参数
            run_py_path = str(Path('model') / 'coupled_0703' / 'run.py')
            cmd = [
                'python', run_py_path,
                '--solution_data_path', tmp_json_path,
                '--resource_path', str(resource_path),
                '--simulation_name', str(simulation_name),
            ]
            if step is not None:
                cmd += ['--step', str(step)]

            # 启动run.py子进程
            run_proc = subprocess.Popen(cmd)
            procs = {'run': run_proc}

            if simulation_address:
                monitor = ResultMonitor(resource_path, simulation_address, solution_name, simulation_name)
                monitor_proc = multiprocessing.Process(target=monitor.run)
                monitor_proc.start()
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
            # 清理管理器
            if key in self.managers:
                try:
                    self.managers[key].shutdown()
                except:
                    pass
                del self.managers[key]
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
            # 清理所有管理器
            for key, manager in list(self.managers.items()):
                try:
                    manager.shutdown()
                except:
                    pass
            self.managers.clear()

    # 可以扩展 rollback, pause, resume 等方法，参数同理加上 key

    def create_shared_memory(self, key):
        # 为每个进程组创建独立的管理器
        manager = multiprocessing.Manager()
        self.managers[key] = manager
        return {
            '1d_data': manager.dict(),       # 模型输出数据
            '2d_data': manager.dict(),
            '1d_ready': manager.Event(),
            '2d_ready': manager.Event(),
            'lock': manager.Lock(),
        }

simulation_process_manager = SimulationProcessManager()