import multiprocessing
from Flood_new import run_flood  #
from pipe_NH import run_pipe_simulation  #

def create_shared_memory():
    manager = multiprocessing.Manager()
    return {
        '1d_data': manager.dict(),       # 模型输出数据
        '2d_data': manager.dict(),
        '1d_ready': manager.Event(),
        '2d_ready': manager.Event(),
        'lock': manager.Lock(),
        'control_signal': manager.Value('i', 0),
        'restart_data': manager.Value('i', 0),
        'flood_process': None,
        'pipe_process': None,
    }

def start_processes(shared):
    if shared['flood_process'] is not None and shared['flood_process'].is_alive():
        return
    if shared['pipe_process'] is not None and shared['pipe_process'].is_alive():
        return

    shared['flood_process'] = multiprocessing.Process(target=run_flood, args=(shared,))
    shared['pipe_process'] = multiprocessing.Process(target=run_pipe_simulation, args=(shared,))

    with shared['lock']:
        shared['control_signal'].value = 0
        shared['restart_data'].value = 0

    # 启动进程
    shared['flood_process'].start()
    shared['pipe_process'].start()
    print("Processes started.")


def stop_processes(shared):
    with shared['lock']:
        shared['control_signal'].value = 1
    if shared['flood_process'] and shared['flood_process'].is_alive():
        shared['flood_process'].join()
    if shared['pipe_process'] and shared['pipe_process'].is_alive():
        shared['pipe_process'].join()
    shared['flood_process'] = None
    shared['pipe_process'] = None
    print("Processes stopped.")


def restart_processes(shared, time_index):
    if shared['flood_process'] is not None and shared['flood_process'].is_alive():
        return
    if shared['pipe_process'] is not None and shared['pipe_process'].is_alive():
        return

    shared['flood_process'] = multiprocessing.Process(target=run_flood, args=(shared,))
    shared['pipe_process'] = multiprocessing.Process(target=run_pipe_simulation, args=(shared,))

    with shared['lock']:
        shared['control_signal'].value = 2
        shared['restart_data'].value = time_index

    # 启动进程
    shared['flood_process'].start()
    shared['pipe_process'].start()
    print("Processes started.")

# import multiprocessing
# from Flood_new import run_flood
# from pipe_NH import run_pipe_simulation
#
#
# class ModelController:
#     def __init__(self):
#         self.shared = self.create_shared_memory()
#
#     def create_shared_memory(self):
#         manager = multiprocessing.Manager()
#         return {
#             '1d_data': manager.dict(),
#             '2d_data': manager.dict(),
#             '1d_ready': manager.Event(),
#             '2d_ready': manager.Event(),
#             'lock': manager.Lock(),
#             'control_signal': manager.Value('i', 0),
#             'restart_data': manager.Value('i', 0),
#             'flood_process': None,
#             'pipe_process': None,
#         }
#
#     def start(self):
#         if self._is_any_process_running():
#             print("Processes already running.")
#             return
#
#         self.shared['flood_process'] = multiprocessing.Process(target=run_flood, args=(self.shared,))
#         self.shared['pipe_process'] = multiprocessing.Process(target=run_pipe_simulation, args=(self.shared,))
#
#         with self.shared['lock']:
#             self.shared['control_signal'].value = 0
#             self.shared['restart_data'].value = 0
#
#         self.shared['flood_process'].start()
#         self.shared['pipe_process'].start()
#         print("Processes started.")
#
#     def stop(self):
#         with self.shared['lock']:
#             self.shared['control_signal'].value = 1
#
#         if self.shared['flood_process'] and self.shared['flood_process'].is_alive():
#             self.shared['flood_process'].join()
#
#         if self.shared['pipe_process'] and self.shared['pipe_process'].is_alive():
#             self.shared['pipe_process'].join()
#
#         self.shared['flood_process'] = None
#         self.shared['pipe_process'] = None
#         print("Processes stopped.")
#
#     def restart(self, time_index=0):
#         self.stop()
#         # 重启前确保进程引用被清理
#         self.shared['flood_process'] = None
#         self.shared['pipe_process'] = None
#
#         self.shared['flood_process'] = multiprocessing.Process(target=run_flood, args=(self.shared,))
#         self.shared['pipe_process'] = multiprocessing.Process(target=run_pipe_simulation, args=(self.shared,))
#
#         with self.shared['lock']:
#             self.shared['control_signal'].value = 2
#             self.shared['restart_data'].value = time_index
#
#         self.shared['flood_process'].start()
#         self.shared['pipe_process'].start()
#         print(f"Processes restarted at time index {time_index}.")
#
#     def _is_any_process_running(self):
#         fp = self.shared['flood_process']
#         pp = self.shared['pipe_process']
#         return (fp is not None and fp.is_alive()) or (pp is not None and pp.is_alive())