# 结果监控功能说明

## 概述

在接收到 START 信号后，系统现在会自动启动一个守护进程来监控结果文件。当检测到结果文件写入完成（通过`.done`文件标记）时，会自动调用 ISimulation 接口将结果返回。

## 功能特性

### 1. 自动文件监控

- 监控指定的 resource 目录
- 每秒检查一次是否有新的`.done`文件
- 支持多文件并发处理

### 2. 结果文件处理

- 自动解析 JSON 格式的结果文件
- 提取 step 信息（从文件名或内容中）
- 转换为 GridResult 对象列表
- 处理高亮网格信息

### 3. 接口调用

- 自动连接 ISimulation 接口
- 调用`send_result`方法发送结果
- 异常处理和日志记录

## 文件格式要求

### 结果文件格式

结果文件应为 JSON 格式，包含以下结构：

```json
{
  "grid_results": [
    {
      "grid_id": 1,
      "water_level": 1.5,
      "u": 0.1,
      "v": 0.2,
      "depth": 0.8
    }
  ],
  "highlight_grids": [1, 5, 10],
  "step": 123
}
```

### 文件命名规则

- 结果文件：`result_step_123.json`
- 完成标记：`result_step_123.json.done`

## 使用方法

### 1. 启动模拟

当发送 START 信号时，系统会自动：

1. 启动模拟进程
2. 启动结果监控器
3. 开始监控结果文件

```python
# 在model.py中已经集成
ok = simulation_process_manager.start(
    solution_name,
    simulation_name,
    resource_path,
    simulation_address
)
```

### 2. 写入结果文件

模拟进程需要：

1. 将结果写入 JSON 文件
2. 创建对应的`.done`文件标记完成

```python
# 示例：写入结果文件
result_data = {
    "grid_results": grid_results,
    "highlight_grids": highlight_grids,
    "step": current_step
}

with open(result_file_path, 'w') as f:
    json.dump(result_data, f)

# 创建完成标记
with open(result_file_path + ".done", 'w') as f:
    f.write("done")
```

### 3. 停止监控

当发送 STOP 信号时，系统会自动：

1. 停止模拟进程
2. 停止结果监控器

## 配置说明

### 监控参数

- 检查间隔：1 秒
- 异常重试间隔：5 秒
- 线程超时：5 秒

### 文件解析

- 支持从文件名提取 step 信息
- 支持从文件内容提取 step 信息
- 自动处理 GridResult 对象转换

## 错误处理

### 常见错误

1. **文件不存在**：跳过处理，继续监控
2. **JSON 解析错误**：记录错误，跳过文件
3. **接口调用失败**：记录错误，继续处理其他文件
4. **网络连接问题**：自动重试，记录错误

### 日志输出

- 处理成功的文件会输出确认信息
- 错误信息会详细记录
- 监控线程异常会记录并继续运行

## 测试

运行测试脚本验证功能：

```bash
python test_result_monitor.py
```

测试脚本会：

1. 创建模拟的结果文件
2. 启动监控器
3. 验证文件处理流程
4. 清理测试文件

## 注意事项

1. **文件权限**：确保监控器有读取 resource 目录的权限
2. **网络连接**：确保能够连接到 ISimulation 接口
3. **文件格式**：严格按照 JSON 格式要求写入结果文件
4. **并发处理**：支持多个结果文件同时处理
5. **资源清理**：停止时会自动清理所有监控线程

## 扩展功能

### 自定义文件格式

可以通过修改`_parse_result_data`方法来支持不同的文件格式。

### 自定义 step 提取

可以通过修改`_extract_step_from_filename`方法来支持不同的文件命名规则。

### 添加更多处理逻辑

可以在`_process_result_file`方法中添加额外的处理逻辑，如数据验证、转换等。
