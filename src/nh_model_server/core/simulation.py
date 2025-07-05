import subprocess
import multiprocessing
import threading
from pathlib import Path
from src.nh_model_server.core.monitor import ResultMonitor

from model.coupled_0703.Flood_new import run_flood
from model.coupled_0703.pipe_NH import run_pipe_simulation

class SimulationProcessManager:
    def __init__(self):
        self.processes = {}  # key: (solution_name, simulation_name), value: process
        self.lock = threading.Lock()

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
            shared = self.create_shared_memory()
            # 启动两个进程
            flood_proc = multiprocessing.Process(
                target=run_flood,
                args=(shared, solution_data, resource_path, step, 1)
            )
            pipe_proc = multiprocessing.Process(
                target=run_pipe_simulation,
                args=(shared, inp_path, resource_path, step)
            )
            flood_proc.start()
            pipe_proc.start()
            # monitor进程
            procs = {}
            procs['flood'] = flood_proc
            procs['pipe'] = pipe_proc
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

    # 可以扩展 rollback, pause, resume 等方法，参数同理加上 key

    def create_shared_memory(self):
        manager = multiprocessing.Manager()
        return {
            '1d_data': manager.dict(),       # 模型输出数据
            '2d_data': manager.dict(),
            '1d_ready': manager.Event(),
            '2d_ready': manager.Event(),
            'lock': manager.Lock(),
        }

simulation_process_manager = SimulationProcessManager()