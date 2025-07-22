# 暂停和恢复功能使用指南

## 概述

暂停和恢复功能允许在模拟运行过程中临时停止子进程，同时保持主监控进程运行。在暂停期间，前端可以添加新的 actions，恢复时这些 actions 会被自动应用。

## 功能特点

### 暂停功能

- **子进程停止**: 所有计算子进程被终止，释放计算资源
- **监控继续**: 主监控进程继续运行，可以接收新的 actions
- **状态保持**: 当前步骤和已完成步骤状态保持不变
- **结果监控**: 仍可轮询获取已完成步骤的结果

### 恢复功能

- **数据刷新**: 重新解析所有数据文件
- **Actions 应用**: 自动应用暂停期间添加的所有新 actions
- **子进程重启**: 使用最新数据重新启动所有子进程
- **状态同步**: 从暂停点继续执行

## API 接口

### 1. 暂停模拟

**POST** `/model/pause_simulation`

**请求体：**

```json
{
  "solution_name": "solution0722",
  "simulation_name": "simulation0722"
}
```

**响应：**

```json
{
  "result": "paused"
}
```

### 2. 恢复模拟

**POST** `/model/resume_simulation`

**请求体：**

```json
{
  "solution_name": "solution0722",
  "simulation_name": "simulation0722",
  "simulation_address": "tcp://localhost:8001"
}
```

**响应：**

```json
{
  "result": "resumed"
}
```

## 使用场景

### 场景 1：动态添加防护措施

```python
import requests
import time

def pause_and_add_actions_example():
    api_base = "http://localhost:8000"

    # 1. 暂停模拟
    pause_response = requests.post(f"{api_base}/model/pause_simulation", json={
        "solution_name": "solution0722",
        "simulation_name": "simulation0722"
    })
    print(f"暂停结果: {pause_response.json()}")

    # 2. 添加新的actions
    # 比如在某个位置添加防洪堤
    add_fence_action = {
        "solution_name": "solution0722",
        "simulation_name": "simulation0722",
        "simulation_address": "tcp://localhost:8001",
        "step": 5,  # 在第5步添加
        "action": {
            "action_type": "add_fence",
            "params": {
                "elevation_delta": 2.0,
                "landuse_type": "fence",
                "feature": {
                    "type": "Polygon",
                    "coordinates": [[[x1, y1], [x2, y2], [x3, y3], [x1, y1]]]
                }
            }
        }
    }

    action_response = requests.post(f"{api_base}/model/add_human_action", json=add_fence_action)
    print(f"添加action结果: {action_response.json()}")

    # 3. 恢复模拟（会自动应用新的action）
    resume_response = requests.post(f"{api_base}/model/resume_simulation", json={
        "solution_name": "solution0722",
        "simulation_name": "simulation0722",
        "simulation_address": "tcp://localhost:8001"
    })
    print(f"恢复结果: {resume_response.json()}")
```

### 场景 2：批量添加多个措施

```python
def batch_add_measures_example():
    api_base = "http://localhost:8000"

    # 暂停模拟
    requests.post(f"{api_base}/model/pause_simulation", json={
        "solution_name": "solution0722",
        "simulation_name": "simulation0722"
    })

    # 批量添加多个防护措施
    measures = [
        {
            "step": 3,
            "action_type": "add_fence",
            "params": {"elevation_delta": 1.5, "landuse_type": "fence", "feature": {...}}
        },
        {
            "step": 4,
            "action_type": "add_gate",
            "params": {"ud_stream": 1, "gate_height": 3, "feature": {...}}
        },
        {
            "step": 5,
            "action_type": "transfer_water",
            "params": {"from_grid": 100, "to_grid": 200, "q": 50.0}
        }
    ]

    # 逐个添加措施
    for measure in measures:
        add_action_payload = {
            "solution_name": "solution0722",
            "simulation_name": "simulation0722",
            "simulation_address": "tcp://localhost:8001",
            "step": measure["step"],
            "action": {
                "action_type": measure["action_type"],
                "params": measure["params"]
            }
        }
        requests.post(f"{api_base}/model/add_human_action", json=add_action_payload)

    # 恢复模拟
    requests.post(f"{api_base}/model/resume_simulation", json={
        "solution_name": "solution0722",
        "simulation_name": "simulation0722",
        "simulation_address": "tcp://localhost:8001"
    })
```

### 场景 3：状态监控和智能恢复

```python
def smart_pause_resume_example():
    api_base = "http://localhost:8000"

    def get_simulation_status():
        response = requests.post(f"{api_base}/model/get_simulation_status", json={
            "solution_name": "solution0722",
            "simulation_name": "simulation0722",
            "simulation_address": "tcp://localhost:8001"
        })
        return response.json()

    def wait_for_step_completion(target_step):
        """等待特定步骤完成"""
        while True:
            status = get_simulation_status()
            if status["result"] == "success":
                current_step = status["status"]["current_step"]
                completed_steps = status["status"]["completed_steps"]

                if target_step in completed_steps or current_step > target_step:
                    print(f"步骤 {target_step} 已完成")
                    break
            time.sleep(2)

    # 等待步骤3完成后暂停
    wait_for_step_completion(3)

    print("步骤3已完成，开始暂停模拟...")
    requests.post(f"{api_base}/model/pause_simulation", json={
        "solution_name": "solution0722",
        "simulation_name": "simulation0722"
    })

    # 分析步骤3的结果，决定是否需要添加措施
    result_response = requests.post(f"{api_base}/model/get_step_result", json={
        "solution_name": "solution0722",
        "simulation_name": "simulation0722",
        "simulation_address": "tcp://localhost:8001",
        "step": 3
    })

    if result_response.json()["result"] == "success":
        result_data = result_response.json()["data"]
        # 根据结果决定是否添加措施
        if needs_intervention(result_data):  # 自定义判断函数
            add_emergency_measures()

    # 恢复模拟
    print("恢复模拟...")
    requests.post(f"{api_base}/model/resume_simulation", json={
        "solution_name": "solution0722",
        "simulation_name": "simulation0722",
        "simulation_address": "tcp://localhost:8001"
    })

def needs_intervention(result_data):
    """分析结果数据，判断是否需要干预"""
    # 实现自定义的判断逻辑
    # 例如：检查水位是否超过警戒线
    return False

def add_emergency_measures():
    """添加紧急措施"""
    # 实现紧急措施添加逻辑
    pass
```

## 状态变化说明

### 暂停前状态

```json
{
  "running": true,
  "paused": false,
  "current_step": 3,
  "monitor_thread_alive": true,
  "child_processes_count": 2,
  "completed_steps": [1, 2],
  "pending_results_count": 2
}
```

### 暂停后状态

```json
{
  "running": true,
  "paused": true,
  "current_step": 3,
  "monitor_thread_alive": true,
  "child_processes_count": 0, // 子进程已停止
  "completed_steps": [1, 2],
  "pending_results_count": 2
}
```

### 恢复后状态

```json
{
  "running": true,
  "paused": false,
  "current_step": 3,
  "monitor_thread_alive": true,
  "child_processes_count": 2, // 子进程重新启动
  "completed_steps": [1, 2],
  "pending_results_count": 2
}
```

## 内部实现逻辑

### 暂停过程

1. 接收暂停请求
2. 终止所有子进程（terminate -> kill if needed）
3. 设置`paused = True`
4. 保持监控线程运行
5. 继续监控结果文件和接收 actions

### 恢复过程

1. 接收恢复请求
2. 重新调用`refresh_and_parse_data()`刷新数据
3. 解析器自动加载和应用所有 actions（包括新添加的）
4. 重新启动所有子进程，传入更新后的数据
5. 设置`paused = False`
6. 继续正常执行

### 数据一致性保证

- **Actions 应用顺序**: 按时间戳排序应用
- **数据状态**: 每次恢复时重新解析确保一致性
- **进程隔离**: 子进程重启避免状态污染
- **监控连续性**: 主线程不中断确保监控完整

## 注意事项

1. **资源管理**: 暂停会释放计算资源，适合长时间中断
2. **状态保持**: 暂停期间仍可轮询已完成的结果
3. **Actions 时序**: 新添加的 actions 会按时间戳正确排序
4. **数据完整性**: 恢复时会重新解析所有数据确保一致性
5. **进程重启**: 恢复时子进程完全重新创建，避免状态残留

## 错误处理

- 如果暂停失败，检查模拟是否正在运行
- 如果恢复失败，检查进程组配置是否完整
- 解析器错误会记录但不阻止进程启动
- 子进程启动失败会有详细错误日志

## 最佳实践

1. **合理时机暂停**: 在步骤完成后暂停，避免中断计算
2. **批量添加措施**: 一次暂停添加多个相关措施
3. **状态监控**: 暂停前后检查状态确认成功
4. **异常处理**: 实现适当的错误重试机制
5. **日志监控**: 关注暂停恢复过程的日志输出

