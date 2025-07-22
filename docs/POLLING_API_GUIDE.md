# 结果轮询 API 接口使用指南

## 概述

本文档介绍了新增的结果轮询 API 接口，这些接口允许前端主动获取模拟结果，而不是被动接收。

## API 端点列表

### 1. 获取已完成步骤列表

**POST** `/model/get_completed_steps`

获取所有已完成但未被拉取的步骤列表。

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
  "result": "success",
  "completed_steps": [1, 2, 3],
  "count": 3
}
```

### 2. 获取指定步骤的结果数据

**POST** `/model/get_step_result`

获取指定步骤的结果数据，获取后该步骤将被标记为已拉取。

**请求体：**

```json
{
  "solution_name": "solution0722",
  "simulation_name": "simulation0722",
  "simulation_address": "tcp://localhost:8001",
  "step": 1
}
```

**成功响应：**

```json
{
  "result": "success",
  "data": {
    "step": 1,
    "data": {
      "flood": ["line1", "line2", "line3"],
      "pipe": ["data1", "data2"]
    },
    "file_types": ["flood", "pipe"],
    "file_suffix": {
      "flood": ".txt",
      "pipe": ".dat"
    },
    "timestamp": "2025-07-22T10:30:45.123456"
  }
}
```

**步骤未就绪响应：**

```json
{
  "result": "not_ready",
  "message": "Step 1 is not ready or already retrieved"
}
```

### 3. 检查步骤是否就绪

**POST** `/model/check_step_ready`

检查指定步骤是否已完成并可以拉取结果。

**请求体：**

```json
{
  "solution_name": "solution0722",
  "simulation_name": "simulation0722",
  "simulation_address": "tcp://localhost:8001",
  "step": 1
}
```

**响应：**

```json
{
  "result": "success",
  "step": 1,
  "ready": true
}
```

### 4. 获取模拟状态

**POST** `/model/get_simulation_status`

获取完整的模拟状态信息，包括已完成步骤统计。

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
  "result": "success",
  "status": {
    "running": true,
    "paused": false,
    "current_step": 4,
    "monitor_thread_alive": true,
    "child_processes_count": 2,
    "completed_steps": [1, 2, 3],
    "pending_results_count": 3
  }
}
```

## 使用示例

### Python 客户端示例

```python
import requests
import time
import json

class SimulationClient:
    def __init__(self, api_base_url, solution_name, simulation_name, simulation_address):
        self.api_base_url = api_base_url
        self.common_payload = {
            "solution_name": solution_name,
            "simulation_name": simulation_name,
            "simulation_address": simulation_address
        }

    def get_completed_steps(self):
        """获取已完成步骤列表"""
        response = requests.post(
            f"{self.api_base_url}/model/get_completed_steps",
            json=self.common_payload
        )
        return response.json()

    def get_step_result(self, step):
        """获取指定步骤的结果"""
        payload = self.common_payload.copy()
        payload["step"] = step
        response = requests.post(
            f"{self.api_base_url}/model/get_step_result",
            json=payload
        )
        return response.json()

    def check_step_ready(self, step):
        """检查步骤是否就绪"""
        payload = self.common_payload.copy()
        payload["step"] = step
        response = requests.post(
            f"{self.api_base_url}/model/check_step_ready",
            json=payload
        )
        return response.json()

    def get_simulation_status(self):
        """获取模拟状态"""
        response = requests.post(
            f"{self.api_base_url}/model/get_simulation_status",
            json=self.common_payload
        )
        return response.json()

def poll_simulation_results():
    """轮询示例"""
    client = SimulationClient(
        api_base_url="http://localhost:8000",
        solution_name="solution0722",
        simulation_name="simulation0722",
        simulation_address="tcp://localhost:8001"
    )

    processed_steps = set()

    while True:
        try:
            # 获取状态
            status_response = client.get_simulation_status()
            if status_response["result"] != "success":
                print(f"获取状态失败: {status_response}")
                time.sleep(5)
                continue

            status = status_response["status"]
            print(f"当前状态: 运行={status['running']}, 当前步骤={status['current_step']}, 待处理结果={status['pending_results_count']}")

            # 如果模拟已停止且没有待处理结果，退出轮询
            if not status['running'] and status['pending_results_count'] == 0:
                print("模拟已完成，所有结果已处理")
                break

            # 获取已完成的步骤
            completed_response = client.get_completed_steps()
            if completed_response["result"] == "success":
                completed_steps = completed_response["completed_steps"]

                # 处理新完成的步骤
                for step in completed_steps:
                    if step not in processed_steps:
                        print(f"发现新完成的步骤: {step}")

                        # 获取结果数据
                        result_response = client.get_step_result(step)
                        if result_response["result"] == "success":
                            result_data = result_response["data"]
                            print(f"成功获取步骤 {step} 的结果:")
                            print(f"  - 文件类型: {result_data['file_types']}")
                            print(f"  - 时间戳: {result_data['timestamp']}")

                            # 处理结果数据
                            process_result_data(step, result_data)
                            processed_steps.add(step)
                        else:
                            print(f"获取步骤 {step} 结果失败: {result_response}")

        except Exception as e:
            print(f"轮询过程中出现错误: {e}")

        # 等待一段时间后继续轮询
        time.sleep(3)

def process_result_data(step, result_data):
    """处理结果数据的示例函数"""
    print(f"处理步骤 {step} 的数据...")

    data = result_data['data']
    file_types = result_data['file_types']

    for file_type in file_types:
        if file_type in data:
            file_data = data[file_type]
            if isinstance(file_data, list):
                print(f"  - {file_type}: {len(file_data)} 行数据")
            else:
                print(f"  - {file_type}: 二进制数据，大小 {len(file_data)} 字节")

    # 这里可以添加具体的数据处理逻辑
    # 例如：保存到数据库、生成可视化图表、发送到其他系统等

if __name__ == "__main__":
    poll_simulation_results()
```

### JavaScript/前端示例

```javascript
class SimulationClient {
  constructor(apiBaseUrl, solutionName, simulationName, simulationAddress) {
    this.apiBaseUrl = apiBaseUrl;
    this.commonPayload = {
      solution_name: solutionName,
      simulation_name: simulationName,
      simulation_address: simulationAddress,
    };
  }

  async getCompletedSteps() {
    const response = await fetch(
      `${this.apiBaseUrl}/model/get_completed_steps`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(this.commonPayload),
      }
    );
    return await response.json();
  }

  async getStepResult(step) {
    const payload = { ...this.commonPayload, step };
    const response = await fetch(`${this.apiBaseUrl}/model/get_step_result`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    return await response.json();
  }

  async getSimulationStatus() {
    const response = await fetch(
      `${this.apiBaseUrl}/model/get_simulation_status`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(this.commonPayload),
      }
    );
    return await response.json();
  }
}

async function pollSimulationResults() {
  const client = new SimulationClient(
    "http://localhost:8000",
    "solution0722",
    "simulation0722",
    "tcp://localhost:8001"
  );

  const processedSteps = new Set();

  while (true) {
    try {
      // 获取状态
      const statusResponse = await client.getSimulationStatus();
      if (statusResponse.result !== "success") {
        console.log("获取状态失败:", statusResponse);
        await new Promise((resolve) => setTimeout(resolve, 5000));
        continue;
      }

      const status = statusResponse.status;
      console.log(
        `当前状态: 运行=${status.running}, 当前步骤=${status.current_step}, 待处理结果=${status.pending_results_count}`
      );

      // 获取并处理新完成的步骤
      const completedResponse = await client.getCompletedSteps();
      if (completedResponse.result === "success") {
        const completedSteps = completedResponse.completed_steps;

        for (const step of completedSteps) {
          if (!processedSteps.has(step)) {
            console.log(`发现新完成的步骤: ${step}`);

            const resultResponse = await client.getStepResult(step);
            if (resultResponse.result === "success") {
              console.log(`成功获取步骤 ${step} 的结果`);
              // 处理结果数据
              processResultData(step, resultResponse.data);
              processedSteps.add(step);
            }
          }
        }
      }

      // 如果模拟已停止且没有待处理结果，退出轮询
      if (!status.running && status.pending_results_count === 0) {
        console.log("模拟已完成，所有结果已处理");
        break;
      }
    } catch (error) {
      console.error("轮询过程中出现错误:", error);
    }

    // 等待3秒后继续轮询
    await new Promise((resolve) => setTimeout(resolve, 3000));
  }
}

function processResultData(step, resultData) {
  console.log(`处理步骤 ${step} 的数据...`);

  const data = resultData.data;
  const fileTypes = resultData.file_types;

  fileTypes.forEach((fileType) => {
    if (data[fileType]) {
      const fileData = data[fileType];
      if (Array.isArray(fileData)) {
        console.log(`  - ${fileType}: ${fileData.length} 行数据`);
      } else {
        console.log(`  - ${fileType}: 二进制数据`);
      }
    }
  });

  // 这里可以添加具体的前端处理逻辑
  // 例如：更新图表、显示在表格中、保存到本地存储等
}

// 启动轮询
pollSimulationResults();
```

## 最佳实践

1. **轮询频率**: 建议轮询间隔为 2-5 秒，避免过于频繁的请求
2. **错误处理**: 实现适当的错误重试机制
3. **状态监控**: 定期检查模拟状态，及时发现异常情况
4. **资源管理**: 及时拉取结果数据，避免内存积压
5. **并发控制**: 避免同时发起多个相同的请求

## 错误码说明

- `success`: 操作成功
- `fail`: 操作失败，查看 error 字段获取详细信息
- `not_ready`: 步骤未就绪或已被拉取（仅用于 get_step_result 接口）

## 注意事项

1. 每个步骤的结果数据只能被拉取一次
2. 拉取后的步骤将从 completed_steps 列表中移除
3. 确保 simulation_address 正确且可访问
4. 大数据量的结果可能需要较长的传输时间

