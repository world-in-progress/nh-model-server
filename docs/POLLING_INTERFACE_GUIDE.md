# 结果轮询接口使用示例

## 概述

修改后的 `Simulation` 类不再主动发送结果，而是提供轮询接口供前端获取结果。

## 新增的方法

### 1. `get_completed_steps() -> list[int]`

获取已完成但未被拉取的步骤列表。

```python
# 示例使用
completed_steps = simulation.get_completed_steps()
print(f"已完成的步骤: {completed_steps}")  # 输出: [1, 2, 3]
```

### 2. `get_step_result(step: int) -> dict[str, Any] | None`

获取指定步骤的结果数据，获取后将该步骤标记为已拉取。

```python
# 示例使用
result = simulation.get_step_result(1)
if result:
    print(f"Step {result['step']} 结果:")
    print(f"数据: {result['data']}")
    print(f"文件类型: {result['file_types']}")
    print(f"时间戳: {result['timestamp']}")
else:
    print("Step 1 尚未完成或已被拉取")
```

### 3. `check_step_ready(step: int) -> bool`

检查指定步骤是否已完成并可以拉取结果。

```python
# 示例使用
if simulation.check_step_ready(2):
    result = simulation.get_step_result(2)
    print("成功获取 step 2 的结果")
else:
    print("Step 2 尚未完成")
```

### 4. `get_status()` (已增强)

获取模拟状态，现在包含已完成步骤的信息。

```python
# 示例使用
status = simulation.get_status()
print(f"运行状态: {status['running']}")
print(f"当前步骤: {status['current_step']}")
print(f"已完成步骤: {status['completed_steps']}")
print(f"待拉取结果数: {status['pending_results_count']}")
```

## 前端轮询示例

```python
import time

def poll_simulation_results(simulation):
    """前端轮询示例"""
    processed_steps = set()

    while simulation.is_running():
        # 获取所有已完成的步骤
        completed_steps = simulation.get_completed_steps()

        for step in completed_steps:
            if step not in processed_steps:
                # 获取结果数据
                result = simulation.get_step_result(step)
                if result:
                    print(f"获取到 Step {step} 的结果:")
                    # 处理结果数据
                    process_result_data(result)
                    processed_steps.add(step)

        # 等待一段时间后再次轮询
        time.sleep(2)

def process_result_data(result):
    """处理结果数据的示例函数"""
    step = result['step']
    data = result['data']
    file_types = result['file_types']

    print(f"处理 Step {step} 的数据...")
    for file_type in file_types:
        if file_type in data:
            print(f"- {file_type}: {len(data[file_type])} 条记录")
```

## 主要变化

1. **监控进程**: 不再主动发送结果，只标记步骤完成状态
2. **结果获取**: 前端需要主动轮询获取结果
3. **状态管理**: 使用 `completed_steps` 集合跟踪已完成但未被拉取的步骤
4. **线程安全**: 所有操作都使用线程锁保护

## 优势

- **解耦合**: 结果生产和消费解耦，提高系统灵活性
- **可控制**: 前端可以控制何时获取结果，避免数据积压
- **状态透明**: 可以随时查询哪些步骤已完成但未被处理
- **防重复**: 每个步骤的结果只能被拉取一次，避免重复处理

