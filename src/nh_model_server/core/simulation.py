import subprocess
import multiprocessing
import threading
from src.nh_model_server.core.monitor import ResultMonitor

class SimulationProcessManager:
    def __init__(self):
        self.processes = {}  # key: (solution_name, simulation_name), value: process
        self.monitors = {}   # key: (solution_name, simulation_name), value: ResultMonitor
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

            # 启动两个进程
            # flood_proc = multiprocessing.Process(
            #     target=run_flood,
            #     args=(solution_name, simulation_name, solution_data, resource_path, step)
            # )
            # pipe_proc = multiprocessing.Process(
            #     target=run_pipe_simulation,
            #     args=(solution_name, simulation_name, solution_data, resource_path, step)
            # )
            # flood_proc.start()
            # pipe_proc.start()
            # self.processes[key] = {'flood': flood_proc, 'pipe': pipe_proc}

            # 启动结果监控器
            if simulation_address:
                monitor = ResultMonitor(resource_path, simulation_address, solution_name, simulation_name)
                monitor.start()
                self.monitors[key] = monitor

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

            # 停止监控器
            monitor = self.monitors.get(key)
            if monitor:
                monitor.stop()
                del self.monitors[key]

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

            # 停止所有监控器
            for key, monitor in list(self.monitors.items()):
                monitor.stop()
            self.monitors.clear()

    # 可以扩展 rollback, pause, resume 等方法，参数同理加上 key

simulation_process_manager = SimulationProcessManager()