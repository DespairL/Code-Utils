# Code-Utils

一个包含各种实用代码工具的Python库，专注于机器学习、深度学习和AI应用开发。

## 模块概览

### 📊 inference_utils - 推理工具
高性能的模型推理工具，支持多种推理引擎和优化配置。

- **SGLang Engine**: 可复用的SGLang推理引擎封装
  - 支持多GPU配置（张量并行、数据并行）
  - 多种推理模式（同步、异步流式、批量推理）
  - 内存优化和量化支持
  - 简单易用的API接口

### 🤖 agent_utils - 智能体工具
构建和管理AI智能体的实用工具。

### 📈 eval_utils - 评估工具
模型评估和基准测试相关的工具集。

### 📊 plot_utils - 可视化工具
数据可视化和图表生成工具。

### 🏋️ training_utils - 训练工具
模型训练相关的实用工具和脚本。

## 快速开始

### 安装依赖

```bash
# 基础依赖
pip install sglang torch transformers

# 可选：用于特定功能
pip install asyncio typing-extensions
```

### 使用示例

#### SGLang推理引擎

```python
from inference_utils.sglang_engine import create_engine

# 创建推理引擎
with create_engine(
    model_path="/path/to/your/model",
    tp_size=1,
    mem_fraction_static=0.7
) as engine:
    # 同步生成
    result = engine.generate("Hello, how are you?")
    print(result)
    
    # 批量推理
    prompts = ["Question 1", "Question 2", "Question 3"]
    results = engine.batch_generate(prompts)
```

#### 异步流式推理

```python
import asyncio
from inference_utils.sglang_engine import SGLangEngine

async def stream_example():
    engine = SGLangEngine(model_path="/path/to/model")
    engine.start_engine()
    
    try:
        async for chunk in engine.async_generate_stream("Tell me a story"):
            print(chunk, end="", flush=True)
    finally:
        engine.shutdown()

asyncio.run(stream_example())
```

## 详细文档

- [SGLang Engine 使用指南](./inference_utils/README.md)
- [示例代码](./inference_utils/example_usage.py)

## 项目结构

```
Code-Utils/
├── README.md                    # 项目主文档
├── .gitignore                   # Git忽略文件
├── agent_utils/                 # 智能体工具
├── eval_utils/                  # 评估工具
├── inference_utils/             # 推理工具
│   ├── README.md               # 模块文档
│   ├── sglang_engine.py        # SGLang引擎封装
│   └── example_usage.py        # 使用示例
├── plot_utils/                  # 可视化工具
└── training_utils/              # 训练工具
```

## 特性

- ✅ **高性能**: 基于SGLang等高性能推理框架
- ✅ **易用性**: 简洁的API设计，支持上下文管理
- ✅ **灵活配置**: 支持多种设备、并行和优化配置
- ✅ **多模式**: 同步、异步、流式、批量等多种推理模式
- ✅ **生产就绪**: 包含错误处理、日志记录等生产特性
- ✅ **文档完善**: 详细的使用文档和示例代码

## 贡献

欢迎提交Issue和Pull Request来改进这个项目！

## 许可证

本项目采用MIT许可证。