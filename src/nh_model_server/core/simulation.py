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
            if key in self.processes and self.processes[key].poll() is None:
                return False  # 该任务已在运行
            
            # 启动模拟进程
            args = ['python', 'test/example_model_entry.py', solution_name, simulation_name, solution_data, resource_path, step]
            if step is not None:
                args.append(str(step))
            proc = subprocess.Popen(args)
            self.processes[key] = proc
            
            # 启动结果监控器
            if simulation_address:
                monitor = ResultMonitor(resource_path, simulation_address, solution_name, simulation_name)
                monitor.start()
                self.monitors[key] = monitor
            
            # mp_proc = multiprocessing.Process(
            #     target=run_simulation,
            #     args=(solution_name, simulation_name, solution_data, resource_path, step)
            # )
            # mp_proc.start()
            # self.processes[key] = mp_proc
            
            return True

    def stop(self, solution_name, simulation_name):
        key = self._get_key(solution_name, simulation_name)
        with self.lock:
            # 停止模拟进程
            proc = self.processes.get(key)
            if proc and proc.poll() is None:
                proc.terminate()
                proc.wait()
                del self.processes[key]
            # # 备用：如果使用 multiprocessing.Process 启动
            # if proc and proc.is_alive():
            #     proc.terminate()
            #     proc.join()
            #     del self.processes[key]
            
            # 停止监控器
            monitor = self.monitors.get(key)
            if monitor:
                monitor.stop()
                del self.monitors[key]
            
            return True

    def stop_all(self):
        """停止所有进程和监控器"""
        with self.lock:
            # 停止所有进程
            for key, proc in list(self.processes.items()):
                if proc and proc.poll() is None:
                    proc.terminate()
                    proc.wait()
            self.processes.clear()
            # # 备用：如果使用 multiprocessing.Process 启动
            # for key, proc in list(self.processes.items()):
            #     if proc and proc.is_alive():
            #         proc.terminate()
            #         proc.join()
            # self.processes.clear()
            
            # 停止所有监控器
            for key, monitor in list(self.monitors.items()):
                monitor.stop()
            self.monitors.clear()

    # 可以扩展 rollback, pause, resume 等方法，参数同理加上 key

simulation_process_manager = SimulationProcessManager()