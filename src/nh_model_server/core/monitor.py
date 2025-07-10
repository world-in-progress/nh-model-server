import os
import time
import c_two as cc
from icrms.isimulation import ISimulation, GridResult

class ResultMonitor:
    """结果文件监控器"""
    def __init__(self, resource_path: str, simulation_address: str, solution_name: str, simulation_name: str, file_types=None, file_suffix=None):
        self.resource_path = resource_path
        self.simulation_address = simulation_address
        self.solution_name = solution_name
        self.simulation_name = simulation_name
        self.running = False
        self.processed_steps = set()  # 已处理过的step
        self.step_files = {}  # step: set of file types 已到达的文件类型
        self.file_types = file_types
        self.file_suffix = file_suffix

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
        """检查结果文件"""
        if not os.path.exists(self.resource_path):
            return
        # 遍历所有.done文件，按step归类
        for filename in os.listdir(self.resource_path):
            if filename.endswith('.done'):
                for file_type in self.file_types:
                    prefix = f'{file_type}_'
                    if filename.startswith(prefix):
                        try:
                            step = int(filename[len(prefix):-5])  # 去掉前缀和.done
                        except Exception:
                            continue
                        if step not in self.step_files:
                            self.step_files[step] = set()
                        self.step_files[step].add(file_type)
                        break
        # 检查哪些step所有类型都到齐且未处理
        for step, types in list(self.step_files.items()):
            if step in self.processed_steps:
                continue
            if all(t in types for t in self.file_types):
                self._process_step(step)
                self.processed_steps.add(step)
    def _process_step(self, step: int):
        """处理某个step的所有结果文件"""
        try:
            # 文件类型与后缀名映射
            result_path = {}
            for file_type in self.file_types:
                suffix = self.file_suffix[file_type] if self.file_suffix and file_type in self.file_suffix else ''
                file_path = os.path.join(self.resource_path, f"{file_type}_{step}{suffix}")
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
