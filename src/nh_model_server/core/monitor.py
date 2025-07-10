import os
import time
import c_two as cc
from icrms.isimulation import ISimulation, GridResult

class ResultMonitor:
    """结果文件监控器"""
    def __init__(self, resource_path: str, simulation_address: str, solution_name: str, simulation_name: str, file_types=None, file_suffix=None, start_step: int = 1):
        self.resource_path = resource_path
        self.simulation_address = simulation_address
        self.solution_name = solution_name
        self.simulation_name = simulation_name
        self.running = False
        self.file_types = file_types
        self.file_suffix = file_suffix
        self.current_step = start_step  # 当前监控的step

    def run(self):
        """以进程方式运行监控循环"""
        self.running = True
        while self.running:
            try:
                self._check_result_files()
                time.sleep(1)  # 每秒检查一次
            except Exception as e:
                print(f"监控进程异常: {e}")
                time.sleep(5)  # 异常时等待更长时间

    def _check_result_files(self):
        """只监测当前step文件夹，发现.done后处理并current_step+1"""
        if not os.path.exists(self.resource_path):
            return
        step_dir = os.path.join(self.resource_path, "step" + str(self.current_step))
        if not os.path.isdir(step_dir):
            return
        done_file = os.path.join(step_dir, '.done')
        if os.path.exists(done_file):
            self._process_step(self.current_step, step_dir)
            print(f"step {self.current_step} 已处理，监控下一个step")
            self.current_step += 1

    def _process_step(self, step: int, step_dir: str):
        """处理某个step文件夹下的所有结果文件"""
        try:
            result_path = {}
            for file_type in self.file_types:
                suffix = self.file_suffix[file_type] if self.file_suffix and file_type in self.file_suffix else ''
                file_path = os.path.join(step_dir, f"{file_type}{suffix}")
                if not os.path.exists(file_path):
                    print(f"缺少数据文件: {file_path}")
                    return
                result_path[file_type] = file_path
            result_data = {}
            for file_type, file_path in result_path.items():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        result_data[file_type] = [line.strip() for line in f if line.strip()]
                except Exception:
                    with open(file_path, 'rb') as f:
                        result_data[file_type] = f.read()
            self._send_result_to_simulation(step, result_data, self.file_types, self.file_suffix)
            print(f"已处理step: {step}")
        except Exception as e:
            print(f"处理step {step} 时出错: {e}")
    
    def _send_result_to_simulation(self, step: int, result_data: dict, file_types: list[str], file_suffix: dict[str, str]):
        """发送结果到ISimulation接口，参数为通用parsed结构"""
        try:
            with cc.compo.runtime.connect_crm(self.simulation_address, ISimulation) as simulation:
                result = simulation.send_result(step, result_data, file_types, file_suffix)
                print(f"结果已发送到simulation, step: {step}, result: {result}")
        except Exception as e:
            print(f"发送结果到simulation时出错: {e}")
