import os
import time
import json
import base64
import threading
import importlib
import c_two as cc
import multiprocessing
from typing import Any
from pathlib import Path
from datetime import datetime
from src.nh_model_server.core.config import settings
from icrms.isimulation import ISimulation, FenceParams, GateParams, TransferWaterParams, ActionType
from persistence.helpers.flood_pipe import get_ne, get_ns, get_rainfall, get_gate, get_tide, apply_add_fence_action, apply_add_gate_action, apply_transfer_water_action

@cc.iicrm
class Simulation(ISimulation):

    def __init__(self, solution_node_key: str, simulation_node_key: str, process_group_config=None):
        self.solution_node_key = solution_node_key
        self.simulation_node_key = simulation_node_key
        self.path = Path(f'{settings.SIMULATION_DIR}{self.simulation_node_key}')
        self.human_action_path = self.path / 'actions' / 'human_action'
        self.result_path = self.path / 'result'
        self.solution_path = Path(f'{settings.SOLUTION_DIR}{self.solution_node_key}')
        self.resource_path = Path(f'{settings.SIMULATION_DIR}{self.simulation_node_key}')

        self.running = False
        self.paused = False  # 暂停状态标志
        self.current_step = 1  # 当前监控的step
        self.process_group_config = process_group_config  # 进程组配置
        self.child_processes = {}  # 子进程字典
        self.manager = None  # multiprocessing.Manager实例
        self.shared = None  # 共享对象字典
        self.model_data = None  # 保存模型数据，用于暂停恢复机制
        
        # 添加线程相关属性
        self.monitor_thread = None  # 监控线程
        self.thread_lock = threading.Lock()  # 线程锁
        
        # 结果轮询相关属性
        self.completed_steps = set()  # 已完成但未被拉取的步骤集合

        self.file_types = process_group_config.get("monitor_config", {}).get("file_types", [])
        self.file_suffix = process_group_config.get("monitor_config", {}).get("file_suffix", {})
        self.model_env = process_group_config.get("model_env", {})

        # Create simulation directory
        self.path.mkdir(parents=True, exist_ok=True)
        self.human_action_path.mkdir(parents=True, exist_ok=True)
        self.result_path.mkdir(parents=True, exist_ok=True)

    def run(self) -> bool:
        """启动监控线程"""
        try:
            with self.thread_lock:
                # 检查是否已经在运行
                if self.monitor_thread and self.monitor_thread.is_alive():
                    print(f"模拟 {self.solution_node_key}_{self.simulation_node_key} 已在运行中")
                    return False
                
                # 如果不是从暂停状态恢复，则读取并保存model_data
                if not self.paused:
                    self.model_data = self._parse_data()

                    # path1 = self.solution_path / 'test' / 'model_input_data.txt'
                    # ne_data = self.model_data.get('ne', {})
                    # ze_list = ne_data.ze_list
                    # under_suf_list = ne_data.under_suf_list
                    # with open(path1, 'w', encoding='utf-8') as f:
                    #     for item in zip(ze_list, under_suf_list):
                    #         f.write(f"{item[0]},{item[1]}\n")

                    # 如果有初始人类行为数据，则解析并更新模型输入数据
                    initial_human_action_path = self.solution_path / 'actions' / 'human_actions'
                    if initial_human_action_path.exists():
                        print(f"找到初始人类行为数据: {initial_human_action_path}")
                        self.model_data = self._update_data(str(initial_human_action_path))

                    # path2 = self.solution_path / 'test' / 'model_input_data_updated.txt'
                    # ne_data_updated = self.model_data.get('ne', {})
                    # ze_list_updated = ne_data_updated.ze_list
                    # under_suf_list_updated = ne_data_updated.under_suf_list
                    # with open(path2, 'w', encoding='utf-8') as f:
                    #     for item in zip(ze_list_updated, under_suf_list_updated):
                    #         f.write(f"{item[0]},{item[1]}\n")

                    if self.model_data is None:
                        print("警告: 初始数据解析失败")
                
                # 创建并启动监控线程
                self.monitor_thread = threading.Thread(
                    target=self._monitor_loop,
                    name=f"monitor_{self.solution_node_key}_{self.simulation_node_key}",
                    daemon=True
                )
                self.monitor_thread.start()
                print(f"模拟 {self.solution_node_key}_{self.simulation_node_key} 监控线程已启动")
                return True
        except Exception as e:
            print(f"启动模拟时出错: {e}")
            import traceback
            traceback.print_exc()
            return False
        
    def stop(self) -> bool:
        """停止监控"""
        with self.thread_lock:
            print(f"收到停止信号，开始停止模拟 {self.solution_node_key}_{self.simulation_node_key}...")
            self.running = False
            self.paused = False  # 重置暂停状态
            
            # 等待监控线程结束
            if self.monitor_thread and self.monitor_thread.is_alive():
                print("等待监控线程结束...")
                self.monitor_thread.join(timeout=10)  # 最多等待10秒
                
                if self.monitor_thread.is_alive():
                    print("警告：监控线程未能在指定时间内结束")
                else:
                    print("监控线程已成功结束")
            
            # 清理子进程（如果线程还没有清理的话）
            self._cleanup_child_processes()
            
            # 清空model_data
            self.model_data = None
            print("model_data已清空")
            
            print(f"模拟 {self.solution_node_key}_{self.simulation_node_key} 已停止")
            return True

    def _parse_data(self) -> dict[str, Any] | None:
        try:
            """解析数据文件，返回模型输入数据"""
            model_input_data = {}
            ne_path = os.path.join(self.solution_path, 'env', self.model_env.get('ne', ''))
            ns_path = os.path.join(self.solution_path, 'env', self.model_env.get('ns', ''))
            rainfall_path = os.path.join(self.solution_path, 'env', self.model_env.get('rainfall', ''))
            gate_path = os.path.join(self.solution_path, 'env', self.model_env.get('gate', ''))
            tide_path = os.path.join(self.solution_path, 'env', self.model_env.get('tide', ''))

            ne_data = get_ne(ne_path) if ne_path else None
            ns_data = get_ns(ns_path) if ns_path else None
            rainfall_data = get_rainfall(rainfall_path) if rainfall_path else None
            gate_data = get_gate(gate_path) if gate_path else None
            tide_data = get_tide(tide_path) if tide_path else None

            model_input_data = {
                "ne": ne_data,
                "ns": ns_data,
                "rainfall": rainfall_data,
                "gate": gate_data,
                "tide": tide_data
            }

            return model_input_data
            
        except Exception as e:
            print(f"解析数据时出错: {e}")
            return None

    def _update_data(self, action_path) -> dict[str, Any] | None:
        """更新模型输入数据 - 遍历指定文件夹中的所有JSON文件"""
        try:
            action_path = Path(action_path)
            
            json_files = list(action_path.glob('*.json'))
            
            if not json_files:
                print(f"在文件夹 {action_path} 中未找到JSON文件")
                return self.model_data
            
            print(f"在文件夹 {action_path} 中找到 {len(json_files)} 个JSON文件")
            
            action_list = []
            # 处理每个JSON文件
            for json_file in json_files:
                print(f"处理JSON文件: {json_file.name}")
                with open(json_file, 'r', encoding='utf-8') as f:
                    action_list.append(json.load(f))

            # 更新模型数据
            for action in action_list:
                action_type = action.get('action_type')
                params = action.get('params', {})

                if action_type == ActionType.ADD_FENCE:
                    self.model_data = apply_add_fence_action(FenceParams(**params), self.model_data)
                elif action_type == ActionType.ADD_GATE:
                    self.model_data = apply_add_gate_action(GateParams(**params), self.model_data)
                elif action_type == ActionType.TRANSFER_WATER:
                    self.model_data = apply_transfer_water_action(TransferWaterParams(**params), self.model_data)
                else:
                    print(f"未知的action_type: {action_type}")

                print(f"已应用动作: {action_type}，参数: {params}")

            return self.model_data
                
        except Exception as e:
            print(f"更新数据时出错: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _monitor_loop(self):
        """以进程方式运行监控循环（在线程中执行）"""
        self.running = True
        try:
            # 启动时创建子进程，使用保存的model_data
            if self.process_group_config:
                self._start_child_processes_with_data(self.model_data)
            
            while self.running:
                try:
                    self._check_result_files()
                    
                    if not self.running:  # 检查是否需要退出
                        break
                    time.sleep(1)  # 每秒检查一次
                except Exception as e:
                    print(f"监控进程异常: {e}")
                    time.sleep(5)  # 异常时等待更长时间
        finally:
            # 清理子进程
            self._cleanup_child_processes()
            # 清理manager
            self._cleanup_manager()
            print(f"Monitor线程 {self.solution_node_key}_{self.simulation_node_key} 已退出")

    def _start_child_processes_with_data(self, model_input_data=None):
        """根据配置创建并启动子进程，使用指定的模型数据"""
        if not self.process_group_config:
            return
        
        try:
            # 获取进程配置
            process_configs = self.process_group_config.get("process_configs", [])
            shared_config = self.process_group_config.get("shared_config", [])
            
            # 在monitor进程内部重新创建共享对象
            shared = self._build_shared_structure(shared_config)
            
            print(f"Monitor正在创建 {len(process_configs)} 个子进程...")
            
            # 如果没有传入模型数据，则使用保存的model_data
            if model_input_data is None:
                model_input_data = self.model_data
            
            # 根据配置创建每个进程
            for proc_config in process_configs:
                proc_name = proc_config["name"]
                script_path = proc_config["script"]
                entrypoint = proc_config["entrypoint"]
                proc_params = proc_config["params"].copy()

                print(proc_params)
                
                # 替换特殊参数
                if "shared" in proc_params and proc_params["shared"] == "shared":
                    proc_params["shared"] = self.shared
                
                if "ne" in proc_params:
                    proc_params['ne'] = model_input_data.get('ne', None)
                    proc_params['ns'] = model_input_data.get('ns', None)
                    proc_params['rainfall'] = model_input_data.get('rainfall', None)
                    proc_params['gate'] = model_input_data.get('gate', None)
                    proc_params['tide'] = model_input_data.get('tide', None)
                
                if "inp" in proc_params:
                    proc_params['inp'] = self.solution_path / 'env' / proc_params['inp']

                # 动态导入模块和函数
                if not os.path.isabs(script_path):
                    module_path = settings.MODEL_PATH + "." + script_path
                else:
                    module_path = script_path
                
                module = importlib.import_module(module_path)
                entrypoint_func = getattr(module, entrypoint)
                
                # 创建并启动进程
                proc = multiprocessing.Process(target=entrypoint_func, kwargs=proc_params)
                print(f"启动子进程: {proc_name}")
                proc.start()
                self.child_processes[proc_name] = proc
                
            print("所有子进程启动完成")
            
        except Exception as e:
            print(f"启动子进程时出错: {e}")
            import traceback
            traceback.print_exc()

    def _build_shared_structure(self, shared_config):
        """在monitor进程内部构建共享对象"""
        if not self.manager:
            self.manager = multiprocessing.Manager()
        
        self.shared = {}
        for item in shared_config:
            name, typ = item["name"], item["type"]
            if typ == "dict":
                self.shared[name] = self.manager.dict()
            elif typ == "Event":
                self.shared[name] = self.manager.Event()
            elif typ == "Lock":
                self.shared[name] = self.manager.Lock()
            else:
                raise ValueError(f"Unsupported shared type: {typ}")
        return self.shared
    
    def _check_result_files(self):
        """检测当前step文件夹和all.done文件"""
        if not os.path.exists(self.result_path):
            return
        
        # 检查all.done文件，如果存在则清理所有进程并退出
        all_done_file = os.path.join(self.result_path, 'all.done')
        if os.path.exists(all_done_file):
            print("检测到all.done文件，开始清理进程组...")
            self._cleanup_child_processes()
            self.running = False
            return
        
        # 检查当前step的.done文件
        step_name = "step" + str(self.current_step)
        step_dir = os.path.join(self.result_path, step_name)
        if not os.path.isdir(step_dir):
            return
        done_file = os.path.join(step_dir, f'{step_name}.done')
        if os.path.exists(done_file):
            print(f"step {self.current_step} 已完成，等待前端轮询获取结果")
            self.current_step += 1

    def _cleanup_child_processes(self):
        """清理所有子进程"""
        if not self.child_processes:
            print("没有子进程需要清理")
            return
        
        try:
            print("Monitor开始清理子进程...")
            for proc_name, proc in self.child_processes.items():
                if proc.is_alive():
                    print(f"正在终止子进程: {proc_name}")
                    proc.terminate()
                    proc.join(timeout=5)  # 等待最多5秒
                    if proc.is_alive():
                        print(f"强制杀死子进程: {proc_name}")
                        proc.kill()
                        proc.join()
                else:
                    print(f"子进程 {proc_name} 已经停止")
            print("所有子进程清理完成")
        except Exception as e:
            print(f"清理子进程时出错: {e}")
    
    def _cleanup_manager(self):
        """清理manager资源"""
        try:
            if self.manager:
                print("清理Manager资源...")
                self.manager.shutdown()
                self.manager = None
                self.shared = None
                print("Manager清理完成")
        except Exception as e:
            print(f"清理Manager时出错: {e}")

    def get_step_result(self, step: int) -> dict[str, Any] | None:
        """获取指定步骤的结果数据，获取后将该步骤标记为已拉取"""
        with self.thread_lock:
            # 读取结果数据
            result_data = self._read_step_result(step)
            
            if result_data is not None:
                print(f"step {step} 结果已被拉取")
                return {
                    'step': step,
                    'data': result_data,
                    'file_types': self.file_types,
                    'file_suffix': self.file_suffix,
                    'timestamp': datetime.now().isoformat()
                }
            
            return None
        
    def _read_step_result(self, step: int) -> dict[str, Any] | None:
        """读取某个step文件夹下的所有结果文件"""
        try:
            step_name = "step" + str(step)
            step_dir = os.path.join(self.result_path, step_name)
            
            # 检查step文件夹是否存在
            if not os.path.isdir(step_dir):
                print(f"step文件夹不存在: {step_dir}")
                return None
                
            # 检查.done文件是否存在
            done_file = os.path.join(step_dir, f'{step_name}.done')
            if not os.path.exists(done_file):
                print(f"step {step} 尚未完成")
                return None
            
            result_path = {}
            for file_type in self.file_types:
                suffix = self.file_suffix[file_type] if self.file_suffix and file_type in self.file_suffix else ''
                file_path = os.path.join(step_dir, f"{file_type}{suffix}")
                if not os.path.exists(file_path):
                    print(f"缺少数据文件: {file_path}")
                    return None
                result_path[file_type] = file_path
                
            result_data = {}
            for file_type, file_path in result_path.items():
                try:
                    # 首先尝试以文本方式读取
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                        # 如果是JSON文件，尝试解析
                        if file_path.endswith('.json'):
                            try:
                                result_data[file_type] = json.loads(content)
                            except json.JSONDecodeError:
                                # JSON解析失败，当作普通文本处理
                                result_data[file_type] = {
                                    'type': 'text',
                                    'content': [line.strip() for line in content.split('\n') if line.strip()]
                                }
                        else:
                            # 普通文本文件，按行分割
                            result_data[file_type] = {
                                'type': 'text',
                                'content': [line.strip() for line in content.split('\n') if line.strip()]
                            }
                except UnicodeDecodeError:
                    # 如果文本读取失败，说明是二进制文件，使用base64编码
                    print(f"文件 {file_path} 为二进制文件，使用base64编码")
                    with open(file_path, 'rb') as f:
                        binary_content = f.read()
                        result_data[file_type] = {
                            'type': 'binary',
                            'content': base64.b64encode(binary_content).decode('utf-8'),
                            'size': len(binary_content)
                        }
                except Exception as e:
                    print(f"读取文件 {file_path} 时出错: {e}")
                    # 最后的兜底方案，尝试二进制读取
                    try:
                        with open(file_path, 'rb') as f:
                            binary_content = f.read()
                            result_data[file_type] = {
                                'type': 'binary',
                                'content': base64.b64encode(binary_content).decode('utf-8'),
                                'size': len(binary_content)
                            }
                    except Exception as binary_error:
                        print(f"二进制读取文件 {file_path} 也失败: {binary_error}")
                        result_data[file_type] = {
                            'type': 'error',
                            'error': str(binary_error)
                        }
                        
            print(f"已读取step {step} 的结果数据")
            return result_data
            
        except Exception as e:
            print(f"读取step {step} 结果时出错: {e}")
            return None
    