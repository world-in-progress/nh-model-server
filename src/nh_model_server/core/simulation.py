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
        self.envs = {}

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

simulation_process_manager = SimulationProcessManager()