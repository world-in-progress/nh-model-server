# 数据解析和 Action 系统架构文档

## 概述

本文档描述了新的数据解析和 action 应用系统的架构。该系统将数据解析逻辑从模型中抽离，统一在持久化层进行管理，支持动态应用 actions 来修改数据。

## 架构组件

### 1. ParserManager (解析器管理器)

- **位置**: `persistence/parser_manager.py`
- **职责**:
  - 管理数据解析过程
  - 加载和应用 actions
  - 将数据转换为模型需要的格式

### 2. Parser Module (解析器模块)

- **位置**: `persistence/parsers/flood_pipe.py`
- **职责**:
  - 提供具体的数据解析函数
  - 提供 action 应用函数
  - 数据格式转换

### 3. Monitor (监控器)

- **位置**: `src/nh_model_server/core/monitor.py`
- **职责**:
  - 在启动/恢复进程时刷新数据
  - 调用解析器管理器处理数据
  - 传递处理后的数据给模型进程

### 4. Process Group Config (进程组配置)

- **位置**: `persistence/process_group.json`
- **职责**:
  - 定义解析器配置
  - 映射数据类型到解析函数
  - 映射 action 类型到处理函数

## 数据流程

```
用户文件路径 → ParserManager → 解析器模块 → 解析数据
                    ↓
Actions文件夹 → 加载Actions → 应用到数据 → 最终模型数据
                    ↓
                Monitor → 模型进程
```

## 配置结构

### 进程组配置格式

```json
{
  "group_type": "flood_pipe",
  "parser_config": {
    "parser_module": "persistence.parsers.flood_pipe",
    "data_parsers": {
      "ne": "get_ne",
      "ns": "get_ns",
      "rainfall": "get_rainfall",
      "tide": "get_tide",
      "gate": "get_gate"
    },
    "action_appliers": {
      "gate_control": "apply_gate_action",
      "rainfall_modification": "apply_rainfall_action",
      "tide_adjustment": "apply_tide_action"
    },
    "model_data_mapping": {
      "ne_data": "ne",
      "ns_data": "ns",
      "rainfall_data": "rainfall",
      "tide_data": "tide",
      "gate_data": "gate"
    }
  },
  "processes": [
    {
      "name": "flood",
      "script": "coupled_0710.Flood_new",
      "entrypoint": "run_flood",
      "parameters": [
        {
          "name": "model_data",
          "type": "dict"
        }
        // ... 其他参数
      ]
    }
  ]
}
```

### Action 文件格式

```json
{
  "timestamp": "2024-07-19T11:00:00Z",
  "type": "gate_control",
  "description": "调整闸门高度",
  "data": {
    "gate_height_changes": [
      {
        "gate_index": 0,
        "new_height": 150
      }
    ]
  }
}
```

## 模型接口变化

### 原来的接口

```python
def run_flood(shared, ne_path, ns_path, inp_path, rainfall_path, gate_path, tide_path, resource_path, step, flag):
    # 在模型内部解析文件
    ne_data = get_ne(ne_path)
    # ...
```

### 新的接口

```python
def run_flood(shared, model_data, resource_path, step, flag):
    # 直接使用解析好的数据
    ne_data = model_data.get("ne_data")
    ns_data = model_data.get("ns_data")
    # ...
```

## 使用方式

### 1. 创建解析器

在 `persistence/parsers/` 目录下创建解析器模块，实现：

- 数据解析函数：`get_xxx(file_path) -> DataClass`
- Action 应用函数：`apply_xxx_action(parsed_data, action) -> parsed_data`

### 2. 配置进程组

在 `process_group.json` 中添加：

- `parser_config`: 解析器配置
- `data_parsers`: 数据类型到解析函数的映射
- `action_appliers`: Action 类型到处理函数的映射
- `model_data_mapping`: 模型参数到数据键的映射

### 3. 创建 Actions

在 solution 的 `actions/` 目录下创建 JSON 文件，定义要应用的修改。

### 4. 模型适配

修改模型入口函数，接受 `model_data` 参数而不是文件路径。

## 优势

1. **解耦**: 数据解析与模型计算分离
2. **复用**: 解析器可在多个模型间复用
3. **动态**: 支持运行时应用 actions 修改数据
4. **统一**: 统一的数据管理和配置方式
5. **可维护**: 清晰的模块边界和职责分工

## 暂停/继续逻辑

1. **暂停**: Monitor 暂停子进程，保持监控状态
2. **继续**: Monitor 重新刷新数据（包括最新 actions），启动新进程
3. **数据同步**: 每次恢复都会自动应用最新的 actions

## 扩展性

- 支持添加新的数据类型和解析器
- 支持添加新的 action 类型
- 支持自定义数据转换逻辑
- 支持多种模型接口适配

