import os
import json
import time
from typing import Dict, Any, Set, Tuple
import c_two as cc
from icrms.isimulation import ISimulation, GridResult

class ResultMonitor:
    """结果文件监控器"""
    def __init__(self, resource_path: str, simulation_address: str, solution_name: str, simulation_name: str):
        self.resource_path = resource_path
        self.simulation_address = simulation_address
        self.solution_name = solution_name
        self.simulation_name = simulation_name
        self.running = False
        self.processed_steps = set()  # 已处理过的step
        self.step_files = {}  # step: set of file types 已到达的文件类型
        self.file_types = ['result', 'flood_nodes', 'hsf']

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

        # 检查哪些step三种类型都到齐且未处理
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
            file_suffix = {
                'result': '.dat',
                'flood_nodes': '.txt',
                'hsf': '.hsf',
            }
            # 读取 result（dat文件为逐行文本，每行一个GridResult，字段顺序：grid_id, water_level, u, v, depth）
            result_file = os.path.join(self.resource_path, f"result_{step}{file_suffix['result']}")
            if not os.path.exists(result_file):
                print(f"缺少数据文件: {result_file}")
                return
            results = []
            with open(result_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            grid_result = GridResult(
                                grid_id=int(parts[0]),
                                water_level=float(parts[1]),
                                u=float(parts[2]),
                                v=float(parts[3]),
                                depth=float(parts[4])
                            )
                            results.append(grid_result)

            # 读取 flood_nodes
            flood_nodes_file = os.path.join(self.resource_path, f"flood_nodes_{step}{file_suffix['flood_nodes']}")
            if not os.path.exists(flood_nodes_file):
                print(f"缺少数据文件: {flood_nodes_file}")
                return
            with open(flood_nodes_file, 'r', encoding='utf-8') as f:
                flood_nodes_data = [int(line.strip()) for line in f if line.strip()]

            # 读取 hsf
            hsf_file = os.path.join(self.resource_path, f"hsf_{step}{file_suffix['hsf']}")
            if not os.path.exists(hsf_file):
                print(f"缺少数据文件: {hsf_file}")
                return
            with open(hsf_file, 'rb') as f:
                hsf_data = f.read()

            self._send_result_to_simulation(step, results, flood_nodes_data, hsf_data)
            print(f"已处理step: {step}")
        except Exception as e:
            print(f"处理step {step} 时出错: {e}")

    def _send_result_to_simulation(self, step: int, results: list[GridResult], flood_nodes_data: list[int], hsf_data: bytes):
        """发送结果到ISimulation接口"""
        try:
            with cc.compo.runtime.connect_crm(self.simulation_address, ISimulation) as simulation:
                # 你可以根据ISimulation的接口定义，调整参数传递方式
                result = simulation.send_result(step, results, flood_nodes_data, hsf_data)
                print(f"结果已发送到simulation, step: {step}, result: {result}")
        except Exception as e:
            print(f"发送结果到simulation时出错: {e}")
