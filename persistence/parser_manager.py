"""
数据解析器管理器
负责根据配置调用相应的解析器，读取数据并应用actions
"""
import os
import json
import importlib
from typing import Dict, Any, Optional, List
from pathlib import Path

class ParserManager:
    """解析器管理器"""
    
    def __init__(self, parser_type: str, parser_config: Dict[str, Any]):
        self.parser_type = parser_type
        self.parser_config = parser_config
        self.parser_module_path = parser_config.get("parser_module")  # 只保存模块路径
        self.parsed_data = {}
        self.actions = []
        
        # 验证解析器模块路径
        if not self.parser_module_path:
            raise ValueError(f"Parser config missing 'parser_module' for type: {self.parser_type}")
    
    def _get_parser_module(self):
        """动态获取解析器模块（每次使用时导入）"""
        try:
            return importlib.import_module(self.parser_module_path)
        except Exception as e:
            print(f"加载解析器模块失败: {e}")
            raise
    
    def parse_data_files(self, file_paths: Dict[str, str]) -> Dict[str, Any]:
        """
        解析数据文件
        
        Args:
            file_paths: 文件路径字典，key为数据类型，value为文件路径
            
        Returns:
            解析后的数据字典
        """
        parser_module = self._get_parser_module()  # 动态获取模块
        
        self.parsed_data = {}
        
        # 根据配置的数据类型和对应的解析函数进行解析
        data_parsers = self.parser_config.get("data_parsers", {})
        
        for data_type, file_path in file_paths.items():
            if data_type in data_parsers:
                parser_func_name = data_parsers[data_type]
                
                # 检查文件是否存在
                if not os.path.exists(file_path):
                    print(f"警告: 数据文件不存在: {file_path}")
                    continue
                
                try:
                    # 获取解析函数
                    parser_func = getattr(parser_module, parser_func_name)
                    
                    # 调用解析函数
                    parsed_result = parser_func(file_path)
                    self.parsed_data[data_type] = parsed_result
                    
                    print(f"成功解析 {data_type} 数据: {file_path}")
                    
                except Exception as e:
                    print(f"解析 {data_type} 数据失败: {e}")
                    raise
            else:
                print(f"警告: 未找到 {data_type} 的解析器配置")
        
        return self.parsed_data
    
    def load_actions(self, solution_path: str) -> List[Dict[str, Any]]:
        """
        加载actions
        
        Args:
            solution_path: solution文件夹路径
            
        Returns:
            actions列表
        """
        actions_dir = os.path.join(solution_path, "actions")
        self.actions = []
        
        if not os.path.exists(actions_dir):
            print(f"Actions目录不存在: {actions_dir}")
            return self.actions
        
        try:
            # 获取所有action文件并按文件名排序
            action_files = []
            for filename in os.listdir(actions_dir):
                if filename.endswith('.json'):
                    action_files.append(filename)
            
            action_files.sort()  # 按文件名排序，确保按时间顺序应用
            
            # 读取所有action文件
            for filename in action_files:
                file_path = os.path.join(actions_dir, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        action = json.load(f)
                        self.actions.append(action)
                        print(f"加载action: {filename}")
                except Exception as e:
                    print(f"加载action文件失败 {filename}: {e}")
            
            print(f"总共加载了 {len(self.actions)} 个actions")
            
        except Exception as e:
            print(f"加载actions时出错: {e}")
        
        return self.actions
    
    def apply_actions_to_data(self, actions: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        将actions应用到解析后的数据上
        
        Args:
            actions: 要应用的actions列表，如果为None则使用已加载的actions
            
        Returns:
            应用actions后的数据字典
        """
        if actions is None:
            actions = self.actions
        
        if not self.parsed_data:
            print("警告: 没有解析后的数据可供应用actions")
            return self.parsed_data
        
        # 获取action应用器配置
        action_appliers = self.parser_config.get("action_appliers", {})
        parser_module = self._get_parser_module()  # 动态获取模块
        
        for action in actions:
            action_type = action.get("type")
            if action_type and action_type in action_appliers:
                applier_func_name = action_appliers[action_type]
                
                try:
                    # 获取action应用函数
                    applier_func = getattr(parser_module, applier_func_name)
                    
                    # 应用action到数据
                    self.parsed_data = applier_func(self.parsed_data, action)
                    
                    print(f"成功应用action类型: {action_type}")
                    
                except Exception as e:
                    print(f"应用action失败 {action_type}: {e}")
            else:
                print(f"警告: 未找到action类型 {action_type} 的应用器")
        
        return self.parsed_data
    
    def get_model_input_data(self) -> Dict[str, Any]:
        """
        获取传递给模型的输入数据
        
        Returns:
            格式化后的模型输入数据
        """
        # 根据配置转换数据格式为模型需要的格式
        model_data_mapping = self.parser_config.get("model_data_mapping", {})
        
        model_input = {}
        for model_param, data_key in model_data_mapping.items():
            if data_key in self.parsed_data:
                model_input[model_param] = self.parsed_data[data_key]
            else:
                print(f"警告: 模型参数 {model_param} 对应的数据 {data_key} 不存在")
        
        return model_input
    
    def refresh_data(self, file_paths: Dict[str, str], solution_path: str) -> Dict[str, Any]:
        """
        刷新数据：重新解析文件，加载最新actions，并应用到数据上
        
        Args:
            file_paths: 文件路径字典
            solution_path: solution路径
            
        Returns:
            最终的模型输入数据
        """
        print("开始刷新数据...")
        
        # 1. 重新解析数据文件
        self.parse_data_files(file_paths)
        
        # 2. 加载最新的actions
        # self.load_actions(solution_path)
        
        # 3. 应用actions到数据
        # self.apply_actions_to_data()
        
        # 4. 获取模型输入数据
        model_input = self.get_model_input_data()
        
        print("数据刷新完成")
        return model_input
    
    def reload_and_apply_actions(self, solution_path: str) -> Dict[str, Any]:
        """
        只重新加载最新的actions并应用到现有的解析数据上（不重新解析基础文件）
        
        Args:
            solution_path: solution路径
            
        Returns:
            应用最新actions后的模型输入数据
        """
        try:
            if not self.parsed_data:
                print("警告: 没有解析后的数据可供应用actions")
                return {}
            
            print("开始重新加载最新actions...")
            
            # 1. 重新加载最新的actions
            self.load_actions(solution_path)
            
            # 2. 应用actions到现有数据
            self.apply_actions_to_data()
            
            # 3. 获取模型输入数据
            model_input = self.get_model_input_data()
            
            print("最新actions加载并应用完成")
            return model_input
            
        except Exception as e:
            print(f"重新加载和应用actions时出错: {e}")
            return {}
