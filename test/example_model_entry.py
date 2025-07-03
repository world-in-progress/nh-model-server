"""
示例模拟进程入口文件
展示如何写入结果文件供守护进程监控
"""
import sys
import json
import time
import os
from typing import List, Dict, Any

def create_grid_result(grid_id: int, water_level: float, u: float, v: float, depth: float) -> Dict[str, Any]:
    """创建网格结果数据"""
    return {
        "grid_id": grid_id,
        "water_level": water_level,
        "u": u,
        "v": v,
        "depth": depth
    }

def write_result_file(resource_path: str, step: int, grid_results: List[Dict[str, Any]], highlight_grids: List[int] = None):
    """写入结果文件"""
    if highlight_grids is None:
        highlight_grids = []
    
    # 准备结果数据
    result_data = {
        "grid_results": grid_results,
        "highlight_grids": highlight_grids,
        "step": step,
        "timestamp": time.time()
    }
    
    # 写入结果文件
    result_file = f"result_step_{step}.json"
    result_path = os.path.join(resource_path, result_file)
    
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(result_data, f, indent=2)
    
    print(f"结果文件已写入: {result_path}")
    
    # 创建完成标记文件
    done_file = result_path + ".done"
    with open(done_file, 'w') as f:
        f.write("done")
    
    print(f"完成标记已创建: {done_file}")

def simulate_step(step: int, grid_count: int = 10) -> List[Dict[str, Any]]:
    """模拟一个步骤的计算"""
    print(f"开始计算步骤 {step}...")
    
    # 模拟计算时间
    time.sleep(2)
    
    # 生成模拟的网格结果
    grid_results = []
    for i in range(grid_count):
        # 模拟一些变化的数据
        water_level = 1.0 + (step * 0.1) + (i * 0.05)
        u = 0.1 + (step * 0.01) + (i * 0.002)
        v = 0.2 + (step * 0.015) + (i * 0.003)
        depth = 0.5 + (step * 0.05) + (i * 0.01)
        
        grid_result = create_grid_result(i + 1, water_level, u, v, depth)
        grid_results.append(grid_result)
    
    print(f"步骤 {step} 计算完成")
    return grid_results

def main():
    """主函数"""
    if len(sys.argv) < 4:
        print("用法: python example_model_entry.py <solution_name> <simulation_name> <resource_path> [start_step]")
        sys.exit(1)
    
    solution_name = sys.argv[1]
    simulation_name = sys.argv[2]
    solution_data = sys.argv[3]
    resource_path = sys.argv[4]
    start_step = int(sys.argv[5]) if len(sys.argv) > 5 else 1
    
    print(f"启动模拟进程:")
    print(f"  解决方案: {solution_name}")
    print(f"  模拟名称: {simulation_name}")
    print(f"  资源路径: {resource_path}")
    print(f"  起始步骤: {start_step}")
    
    # 确保资源目录存在
    os.makedirs(resource_path, exist_ok=True)
    
    # 模拟多个步骤
    total_steps = 5
    for step in range(start_step, start_step + total_steps):
        try:
            # 计算当前步骤
            grid_results = simulate_step(step)
            
            # 确定高亮网格（示例：每3步高亮不同的网格）
            highlight_grids = [(step % 3) + 1, (step % 5) + 1]
            
            # 写入结果文件
            write_result_file(resource_path, step, grid_results, highlight_grids)
            
            # 等待一段时间再处理下一步
            time.sleep(1)
            
        except KeyboardInterrupt:
            print("收到中断信号，停止模拟")
            break
        except Exception as e:
            print(f"步骤 {step} 处理出错: {e}")
            continue
    
    print("模拟进程完成")

if __name__ == "__main__":
    main() 