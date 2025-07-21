import os
import time
import multiprocessing
import importlib
import c_two as cc
from icrms.isimulation import ISimulation, GridResult
from src.nh_model_server.core.config import settings
from persistence.parser_manager import ParserManager

class ResultMonitor:
    """结果文件监控器"""
    def __init__(self, resource_path: str, simulation_address: str, solution_name: str, simulation_name: str, 
                 process_group_config=None, start_step: int = 1, stop_event=None,
                 pause_event=None, resume_event=None, update_config_event=None):
        self.resource_path = resource_path
        self.simulation_address = simulation_address
        self.solution_name = solution_name
        self.simulation_name = simulation_name
        self.running = False
        self.paused = False  # 暂停状态标志
        self.current_step = start_step  # 当前监控的step
        self.process_group_config = process_group_config  # 进程组配置
        self.child_processes = {}  # 子进程字典
        self.manager = None  # multiprocessing.Manager实例
        self.shared = None  # 共享对象字典
        self.stop_event = stop_event  # 跨进程停止信号
        self.pause_event = pause_event  # 暂停信号
        self.resume_event = resume_event  # 继续信号
        self.update_config_event = update_config_event  # 配置更新信号

        self.file_types = process_group_config.get("monitor_config", {}).get("file_types", [])
        self.file_suffix = process_group_config.get("monitor_config", {}).get("file_suffix", {})
        self.env = process_group_config.get("env", {})

        # 初始化解析器管理器
        self.parser_manager = None
        self._init_parser_manager()

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

    def run(self):
        """以进程方式运行监控循环"""
        self.running = True
        try:
            # 启动时先创建子进程
            if self.process_group_config:
                self._start_child_processes()
            
            while self.running:
                try:
                    # 检查停止信号
                    if self.stop_event and self.stop_event.is_set():
                        print("Monitor收到停止信号，开始退出...")
                        self.running = False
                        break
                    
                    # 检查暂停信号
                    if self.pause_event and self.pause_event.is_set():
                        print("Monitor收到暂停信号，停止子进程但保持监控运行...")
                        self._pause_child_processes()
                        self.pause_event.clear()  # 清除暂停信号
                    
                    # 检查继续信号
                    if self.resume_event and self.resume_event.is_set():
                        print("Monitor收到继续信号，重新启动子进程...")
                        # 检查是否需要更新配置
                        if self.update_config_event and self.update_config_event.is_set():
                            print("检测到配置更新信号，重新加载进程组配置...")
                            self._update_process_group_config()
                            self.update_config_event.clear()
                        self._resume_child_processes()
                        self.resume_event.clear()  # 清除继续信号
                    
                    # 只有在非暂停状态下才检查结果文件
                    if not self.paused:
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
            print(f"Monitor进程 {self.solution_name}_{self.simulation_name} 已退出")
            # 确保进程能够正常退出
            os._exit(0)

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
    
    def stop(self):
        """停止监控"""
        print("检测到停止信号，开始清理进程组...")
        self._cleanup_child_processes()
        self.running = False
        return

    def _start_child_processes(self):
        """根据配置创建并启动子进程"""
        if not self.process_group_config:
            return
        
        try:
            # 获取进程配置
            process_configs = self.process_group_config.get("process_configs", [])
            shared_config = self.process_group_config.get("shared_config", [])
            
            # 在monitor进程内部重新创建共享对象
            shared = self._build_shared_structure(shared_config)
            
            print(f"Monitor正在创建 {len(process_configs)} 个子进程...")
            
            # 刷新并解析数据
            model_input_data = self.refresh_and_parse_data()
            
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
    
    def _pause_child_processes(self):
        """暂停所有子进程（终止但不清理进程记录）"""
        if not self.child_processes:
            print("没有子进程需要暂停")
            return
        
        try:
            print("Monitor开始暂停子进程...")
            for proc_name, proc in list(self.child_processes.items()):
                if proc.is_alive():
                    print(f"正在暂停子进程: {proc_name}")
                    proc.terminate()
                    proc.join(timeout=5)  # 等待最多5秒
                    if proc.is_alive():
                        print(f"强制杀死子进程: {proc_name}")
                        proc.kill()
                        proc.join()
                    print(f"子进程 {proc_name} 已暂停")
                else:
                    print(f"子进程 {proc_name} 已经停止")
            
            self.paused = True
            print("所有子进程已暂停，Monitor保持运行状态")
        except Exception as e:
            print(f"暂停子进程时出错: {e}")
    
    def _resume_child_processes(self):
        """恢复所有子进程（重新启动）"""
        if not self.process_group_config:
            print("没有进程组配置，无法恢复子进程")
            return
            
        try:
            print("Monitor开始恢复子进程...")
            # 清理旧的进程记录
            self.child_processes.clear()
            
            # 重新启动子进程（会自动刷新数据和应用最新actions）
            self._start_child_processes()
            
            self.paused = False
            print("所有子进程已恢复运行")
        except Exception as e:
            print(f"恢复子进程时出错: {e}")
    
    def _update_process_group_config(self):
        """更新进程组配置"""
        try:
            print("正在更新进程组配置...")
            # 这里可以重新加载配置文件或从外部获取新的配置
            # 暂时保持现有配置，实际使用时可以从某个共享存储中获取新配置
            print("进程组配置更新完成")
        except Exception as e:
            print(f"更新进程组配置时出错: {e}")
    
    def update_config(self, new_config):
        """更新进程组配置的外部接口"""
        self.process_group_config = new_config
        print("进程组配置已更新")
    
    def update_env(self, new_env):
        """更新数据文件路径"""
        self.env = new_env
        print(f"数据文件路径已更新: {new_env}")
