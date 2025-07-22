import os
import time
import json
import importlib
import threading
import c_two as cc
import multiprocessing
from typing import Any
from pathlib import Path
from datetime import datetime
from src.nh_model_server.core.config import settings
from persistence.parser_manager import ParserManager
from icrms.isimulation import FenceParams, GateParams, ISimulation, HumanAction, ActionType

@cc.iicrm
class Simulation(ISimulation):

    def __init__(self, resource_path: str, solution_name: str, simulation_name: str, process_group_config=None):
        
        self.solution_name = solution_name
        self.simulation_name = simulation_name
        self.path = Path(f'{settings.SIMULATION_DIR}{self.simulation_name}')
        self.human_action_path = self.path / 'human_action'
        self.result_path = self.path / 'result'
        self.solution_path = Path(f'{settings.SOLUTION_DIR}{self.solution_name}')
        self.resource_path = resource_path
        
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
        self.env = process_group_config.get("env", {})
        # 初始化解析器管理器
        self.parser_manager = None
        self._init_parser_manager()

        # Create simulation directory
        self.path.mkdir(parents=True, exist_ok=True)
        self.human_action_path.mkdir(parents=True, exist_ok=True)
        self.result_path.mkdir(parents=True, exist_ok=True)
        
    def _init_parser_manager(self):
        """初始化解析器管理器"""
        try:
            if self.process_group_config:
                parser_config = self.process_group_config.get("parser_config")
                if parser_config:
                    group_type = self.process_group_config.get("group_type", "unknown")
                    self.parser_manager = ParserManager(group_type, parser_config)
                    print(f"解析器管理器初始化成功: {group_type}")
                else:
                    print("警告: 进程组配置中没有parser_config")
            else:
                print("警告: 没有进程组配置")
        except Exception as e:
            print(f"初始化解析器管理器失败: {e}")
            self.parser_manager = None

    def refresh_and_parse_data(self):
        """刷新并解析数据，应用最新actions"""
        try:
            if not self.parser_manager:
                print("警告: 解析器管理器未初始化")
                return None
                
            # 构建完整的文件路径
            solution_path = os.path.dirname(self.resource_path)
            full_env = {}
            
            for data_type, file_name in self.env.items():
                if file_name:
                    full_path = os.path.join(solution_path, file_name)
                    full_env[data_type] = full_path
            
            # 刷新数据：解析文件 + 加载actions + 应用actions
            model_input_data = self.parser_manager.refresh_data(full_env, solution_path)
            
            print(f"数据刷新完成，解析了 {len(full_env)} 个文件")
            return model_input_data
            
        except Exception as e:
            print(f"刷新和解析数据时出错: {e}")
            return None

    def _apply_new_actions_to_model_data(self):
        """将最新的actions应用到保存的model_data上"""
        try:
            if not self.parser_manager or self.model_data is None:
                print("警告: 解析器管理器未初始化或model_data为空")
                return None
                
            # 构建solution路径
            solution_path = os.path.dirname(self.resource_path)
            
            # 只重新加载和应用actions，而不重新解析基础文件
            updated_model_data = self.parser_manager.apply_actions_to_data(self.model_data, solution_path)
            
            print("最新actions已应用到保存的model_data")
            return updated_model_data
            
        except Exception as e:
            print(f"应用新actions到model_data时出错: {e}")
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
            print(f"Monitor线程 {self.solution_name}_{self.simulation_name} 已退出")

    def _check_result_files(self):
        """检测当前step文件夹和all.done文件"""
        if not os.path.exists(self.resource_path):
            return
        
        # 检查all.done文件，如果存在则清理所有进程并退出
        all_done_file = os.path.join(self.resource_path, 'all.done')
        if os.path.exists(all_done_file):
            print("检测到all.done文件，开始清理进程组...")
            self._cleanup_child_processes()
            self.running = False
            return
        
        # 检查当前step的.done文件
        step_name = "step" + str(self.current_step)
        step_dir = os.path.join(self.resource_path, step_name)
        if not os.path.isdir(step_dir):
            return
        done_file = os.path.join(step_dir, f'{step_name}.done')
        if os.path.exists(done_file):
            # 只标记步骤完成，不主动发送结果
            self.completed_steps.add(self.current_step)
            print(f"step {self.current_step} 已完成，等待前端轮询获取结果")
            self.current_step += 1

    def _read_step_result(self, step: int) -> dict[str, Any] | None:
        """读取某个step文件夹下的所有结果文件"""
        try:
            step_name = "step" + str(step)
            step_dir = os.path.join(self.resource_path, step_name)
            
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
                    with open(file_path, 'r', encoding='utf-8') as f:
                        result_data[file_type] = [line.strip() for line in f if line.strip()]
                except Exception:
                    with open(file_path, 'rb') as f:
                        result_data[file_type] = f.read()
                        
            print(f"已读取step {step} 的结果数据")
            return result_data
            
        except Exception as e:
            print(f"读取step {step} 结果时出错: {e}")
            return None
    
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
                
                # 如果有model_data参数，传入解析后的数据
                if "model_data" in proc_params:
                    proc_params["model_data"] = model_input_data
                
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
    
    ############# ISimulation接口方法 #############

    def run(self) -> bool:
        """启动监控线程"""
        try:
            with self.thread_lock:
                # 检查是否已经在运行
                if self.monitor_thread and self.monitor_thread.is_alive():
                    print(f"模拟 {self.solution_name}_{self.simulation_name} 已在运行中")
                    return False
                
                # 如果不是从暂停状态恢复，则读取并保存model_data
                if not self.paused:
                    print("首次启动，读取并应用初始actions...")
                    self.model_data = self.refresh_and_parse_data()
                    if self.model_data is None:
                        print("警告: 初始数据解析失败")
                
                # 创建并启动监控线程
                self.monitor_thread = threading.Thread(
                    target=self._monitor_loop,
                    name=f"monitor_{self.solution_name}_{self.simulation_name}",
                    daemon=True
                )
                self.monitor_thread.start()
                print(f"模拟 {self.solution_name}_{self.simulation_name} 监控线程已启动")
                return True
        except Exception as e:
            print(f"启动模拟时出错: {e}")
            import traceback
            traceback.print_exc()
            return False

    def stop(self) -> bool:
        """停止监控"""
        with self.thread_lock:
            print(f"收到停止信号，开始停止模拟 {self.solution_name}_{self.simulation_name}...")
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
            
            print(f"模拟 {self.solution_name}_{self.simulation_name} 已停止")
            return True

    def pause(self) -> bool:
        """暂停模拟（停止监控线程和子进程，但保持model_data）"""
        with self.thread_lock:
            if not self.running:
                print("模拟未运行，无法暂停")
                return False
            
            if self.paused:
                print("模拟已处于暂停状态")
                return True
            
            print(f"开始暂停模拟 {self.solution_name}_{self.simulation_name}...")
            
            # 停止运行标志，这会导致监控线程退出
            self.running = False
            
            # 等待监控线程结束
            if self.monitor_thread and self.monitor_thread.is_alive():
                print("等待监控线程结束...")
                self.monitor_thread.join(timeout=10)
                
                if self.monitor_thread.is_alive():
                    print("警告：监控线程未能在指定时间内结束")
                else:
                    print("监控线程已成功结束")
            
            # 清理子进程和manager
            self._cleanup_child_processes()
            self._cleanup_manager()
            
            # 设置暂停状态，但保持model_data
            self.paused = True
            print(f"模拟 {self.solution_name}_{self.simulation_name} 已暂停，model_data已保留")
            return True

    def resume(self) -> bool:
        """恢复模拟（重新启动监控线程和子进程，应用最新的actions到保存的model_data）"""
        with self.thread_lock:
            if not self.paused:
                print("模拟未处于暂停状态，无法恢复")
                return False
            
            if self.model_data is None:
                print("错误：没有保存的model_data，无法恢复")
                return False
            
            print(f"开始恢复模拟 {self.solution_name}_{self.simulation_name}...")
            
            # 获取最新的actions并应用到保存的model_data上
            print("恢复时重新获取并应用最新actions...")
            updated_model_data = self._apply_new_actions_to_model_data()
            
            if updated_model_data is not None:
                self.model_data = updated_model_data
            else:
                print("警告: 应用新actions失败，将使用原有model_data")
            
            # 重置状态并重新启动监控线程
            self.paused = False
            self.running = True
            
            # 创建并启动监控线程
            self.monitor_thread = threading.Thread(
                target=self._monitor_loop,
                name=f"monitor_{self.solution_name}_{self.simulation_name}",
                daemon=True
            )
            self.monitor_thread.start()
            
            print(f"模拟 {self.solution_name}_{self.simulation_name} 已恢复，actions已应用")
            return True

    def is_running(self):
        """检查模拟是否正在运行（包括暂停状态）"""
        with self.thread_lock:
            # 如果是暂停状态，返回True表示模拟仍然"存在"但暂停
            if self.paused:
                return True
            # 否则检查是否真正在运行
            return (self.running and 
                   self.monitor_thread and 
                   self.monitor_thread.is_alive())

    def get_status(self):
        """获取模拟状态"""
        with self.thread_lock:
            return {
                "running": self.running,
                "paused": self.paused,
                "current_step": self.current_step,
                "monitor_thread_alive": (self.monitor_thread and self.monitor_thread.is_alive()),
                "child_processes_count": len([p for p in self.child_processes.values() 
                                            if hasattr(p, 'is_alive') and p.is_alive()]),
                "completed_steps": list(self.completed_steps),
                "pending_results_count": len(self.completed_steps),
                "has_model_data": self.model_data is not None
            }

    def get_simulation_status(self):
        """获取模拟状态 - ISimulation接口方法"""
        return self.get_status()

    def update_config(self, new_config):
        """更新进程组配置的外部接口"""
        self.process_group_config = new_config
        print("进程组配置已更新")
    
    def update_env(self, new_env):
        """更新数据文件路径"""
        self.env = new_env
        print(f"数据文件路径已更新: {new_env}")

    def get_completed_steps(self) -> list[int]:
        """获取已完成但未被拉取的步骤列表"""
        with self.thread_lock:
            return list(self.completed_steps)
    
    def get_step_result(self, step: int) -> dict[str, Any] | None:
        """获取指定步骤的结果数据，获取后将该步骤标记为已拉取"""
        with self.thread_lock:
            # 检查步骤是否在已完成列表中
            if step not in self.completed_steps:
                return None
            
            # 读取结果数据
            result_data = self._read_step_result(step)
            
            if result_data is not None:
                # 从已完成集合中移除该步骤，表示已被拉取
                self.completed_steps.remove(step)
                print(f"step {step} 结果已被拉取")
                return {
                    'step': step,
                    'data': result_data,
                    'file_types': self.file_types,
                    'file_suffix': self.file_suffix,
                    'timestamp': datetime.now().isoformat()
                }
            
            return None
    
    def check_step_ready(self, step: int) -> bool:
        """检查指定步骤是否已完成并可以拉取结果"""
        with self.thread_lock:
            return step in self.completed_steps


    def get_human_actions(self, step: int) -> list[HumanAction]:
        step_path = self.human_action_path / str(step)
        action_files = step_path.glob('*.json')
        actions = []

        # 按时间排序，基于文件名中的时间戳
        action_files = sorted(action_files, key=lambda x: datetime.strptime(x.stem.split('_')[-1], "%Y-%m-%d-%H-%M-%S-%f"))
        
        for action_file in action_files:
            with open(action_file, 'r', encoding='utf-8') as f:
                action = HumanAction.model_validate_json(f.read())
                actions.append(action)
        
        return actions
       
    def add_human_action(self, step: int, action: HumanAction) -> dict[str, bool | str]:
        try:
            step_path = self.human_action_path / str(step)
            step_path.mkdir(parents=True, exist_ok=True)

            # 使用毫秒级别的时间戳生成唯一时间标识
            time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")
            action_path = step_path / f'action_{time}.json'

            if(action.action_type == ActionType.ADD_FENCE | ActionType.ADD_GATE):
                feature = action.params.feature
                # TODO: 获取 feature 的所有 grid_id
                grid_id_list = []
                if(action.action_type == ActionType.ADD_FENCE):
                    params = FenceParams(action.params.elevation_delta, action.params.landuse_type, grid_id_list)
                else:
                    params = GateParams(action.params.ud_stream, action.params.gate_height, grid_id_list)
                action.params = params
            with open(action_path, 'w', encoding='utf-8') as f:
                json.dump(action.model_dump(), f, ensure_ascii=False, indent=4)

            return {'success': True, 'message': 'success'}
        except Exception as e:
            return {'success': False, 'message': str(e)}
        