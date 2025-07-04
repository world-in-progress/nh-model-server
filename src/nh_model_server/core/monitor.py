import os
import json
import time
from typing import Dict, Any
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
        self.processed_files = set()

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

        for filename in os.listdir(self.resource_path):
            if filename.endswith('.done') and filename not in self.processed_files:
                result_file = filename[:-5]  # 移除.done后缀
                result_path = os.path.join(self.resource_path, result_file)
                
                if os.path.exists(result_path):
                    self._process_result_file(result_path, result_file)
                    self.processed_files.add(filename)

    def _process_result_file(self, result_path: str, result_file: str):
        """处理结果文件"""
        try:
            # 解析文件名获取step信息
            # 假设文件名格式为: result_step_xxx.json 或类似格式
            step = self._extract_step_from_filename(result_file)
            
            # 读取结果文件
            with open(result_path, 'r', encoding='utf-8') as f:
                result_data = json.load(f)
            
            # 转换为GridResult对象列表
            grid_results = self._parse_result_data(result_data)
            
            # 获取高亮网格（如果有的话）
            highlight_grids = result_data.get('highlight_grids', [])
            
            # 调用ISimulation接口发送结果
            self._send_result_to_simulation(step, grid_results, highlight_grids)
            
            print(f"已处理结果文件: {result_file}, step: {step}")
            
        except Exception as e:
            print(f"处理结果文件 {result_file} 时出错: {e}")

    def _extract_step_from_filename(self, filename: str) -> int:
        """从文件名中提取step信息"""
        # 这里需要根据实际的文件命名规则来实现
        # 示例实现，假设文件名包含step信息
        try:
            # 假设文件名格式为: result_step_123.json
            if 'step_' in filename:
                step_part = filename.split('step_')[1].split('.')[0]
                return int(step_part)
            else:
                # 如果没有明确的step信息，使用文件名作为step
                return hash(filename) % 10000  # 简单的哈希值作为step
        except:
            return 0

    def _parse_result_data(self, result_data: Dict[str, Any]) -> list[GridResult]:
        """解析结果数据为GridResult对象列表"""
        grid_results = []
        
        # 根据实际的数据格式来解析
        # 这里假设result_data包含grid_results数组
        if 'grid_results' in result_data:
            for grid_data in result_data['grid_results']:
                grid_result = GridResult(
                    grid_id=grid_data.get('grid_id', 0),
                    water_level=grid_data.get('water_level', 0.0),
                    u=grid_data.get('u', 0.0),
                    v=grid_data.get('v', 0.0),
                    depth=grid_data.get('depth', 0.0)
                )
                grid_results.append(grid_result)
        
        return grid_results

    def _send_result_to_simulation(self, step: int, grid_results: list[GridResult], highlight_grids: list[int]):
        """发送结果到ISimulation接口"""
        try:
            with cc.compo.runtime.connect_crm(self.simulation_address, ISimulation) as simulation:
                result = simulation.send_result(step, grid_results, highlight_grids)
                print(f"结果已发送到simulation, step: {step}, result: {result}")
        except Exception as e:
            print(f"发送结果到simulation时出错: {e}")
