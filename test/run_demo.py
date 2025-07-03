#!/usr/bin/env python3
"""
演示脚本：展示完整的START信号处理流程
包括启动模拟进程和结果监控
"""
import os
import time
import threading
from src.nh_model_server.core.simulation import simulation_process_manager
from src.nh_model_server.core.config import settings

def demo_start_simulation():
    """演示启动模拟"""
    print("=== 演示：START信号处理流程 ===")
    
    # 模拟参数
    solution_name = "demo_solution"
    simulation_name = "demo_simulation"
    simulation_address = "http://localhost:8080/simulation"  # 模拟地址
    
    # 创建资源路径
    resource_path = os.path.join(settings.RESOURCE_PATH, solution_name, simulation_name)
    os.makedirs(resource_path, exist_ok=True)
    
    print(f"资源路径: {resource_path}")
    
    # 启动模拟进程和监控器
    print("启动模拟进程和结果监控器...")
    ok = simulation_process_manager.start(
        solution_name, 
        simulation_name, 
        resource_path, 
        simulation_address
    )
    
    if ok:
        print("✓ 模拟进程和监控器启动成功")
    else:
        print("✗ 模拟进程已在运行")
        return
    
    # 等待一段时间让模拟进程运行
    print("等待模拟进程运行...")
    time.sleep(20)
    
    # 停止模拟
    print("停止模拟进程和监控器...")
    simulation_process_manager.stop(solution_name, simulation_name)
    print("✓ 模拟进程和监控器已停止")
    
    # 清理资源目录
    import shutil
    if os.path.exists(resource_path):
        shutil.rmtree(resource_path)
        print("✓ 资源目录已清理")

def demo_multiple_simulations():
    """演示多个模拟并发运行"""
    print("\n=== 演示：多个模拟并发运行 ===")
    
    simulations = [
        ("solution_1", "sim_1", "http://localhost:8080/sim1"),
        ("solution_2", "sim_2", "http://localhost:8080/sim2"),
        ("solution_3", "sim_3", "http://localhost:8080/sim3"),
    ]
    
    # 启动多个模拟
    for solution_name, simulation_name, sim_address in simulations:
        resource_path = os.path.join(settings.RESOURCE_PATH, solution_name, simulation_name)
        os.makedirs(resource_path, exist_ok=True)
        
        ok = simulation_process_manager.start(
            solution_name, simulation_name, resource_path, sim_address
        )
        
        if ok:
            print(f"✓ 启动 {solution_name}/{simulation_name}")
        else:
            print(f"✗ {solution_name}/{simulation_name} 已在运行")
    
    # 等待运行
    print("等待模拟进程运行...")
    time.sleep(15)
    
    # 停止所有模拟
    print("停止所有模拟...")
    for solution_name, simulation_name, _ in simulations:
        simulation_process_manager.stop(solution_name, simulation_name)
        print(f"✓ 停止 {solution_name}/{simulation_name}")
    
    # 清理
    import shutil
    for solution_name, simulation_name, _ in simulations:
        resource_path = os.path.join(settings.RESOURCE_PATH, solution_name, simulation_name)
        if os.path.exists(resource_path):
            shutil.rmtree(resource_path)

def main():
    """主函数"""
    print("NH Model Server 演示")
    print("=" * 50)
    
    try:
        # 演示单个模拟
        demo_start_simulation()
        
        # 演示多个模拟
        demo_multiple_simulations()
        
        print("\n=== 演示完成 ===")
        
    except KeyboardInterrupt:
        print("\n收到中断信号，停止所有进程...")
        simulation_process_manager.stop_all()
        print("✓ 所有进程已停止")
    except Exception as e:
        print(f"演示过程中出错: {e}")
        simulation_process_manager.stop_all()

if __name__ == "__main__":
    main() 