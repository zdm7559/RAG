# Mini RAG API

这是一个用于学习大模型应用开发的迷你 RAG 项目，当前使用 FastAPI、Qdrant、本地 embedding 模型和本地 rerank 模型搭建了一条可运行的检索增强问答链路。

这个仓库的目标不是一次性做成“完整产品”，而是把 RAG 系统拆成几个清晰阶段，边做边学，逐步往里增加能力，并且始终保持项目可以运行、可以观察、可以调试。

## 项目当前能做什么

目前这个项目已经支持：

- 从本地目录加载 `.txt`、`.md`、`.pdf` 文档
- 对文档做清洗和切块
- 使用本地 embedding 模型生成向量
- 将 chunk 向量和元数据写入本地 Qdrant
- 根据用户问题做向量召回
- 对召回结果进行本地 rerank
- 选择最终上下文并交给 LLM 生成回答
- 返回带引用信息的答案
- 查看已入库文档、查看 chunk、删除文档

## 当前检索流程

当前版本的主流程是：

1. 文档读取
2. 文本清洗
3. 文本切块
4. embedding 检索
5. rerank 重排
6. 上下文筛选
7. LLM 生成答案

当前默认参数为：

- 第一次召回数量：30
- 进入 rerank 数量：10
- 最终送入 LLM 的 chunk 数量：5
- 召回分数阈值：可选
- 重排分数阈值：可选

## 技术栈

- 后端：FastAPI
- 向量数据库：Qdrant（本地模式）
- Embedding 模型：`BAAI/bge-small-en-v1.5`
- Rerank 模型：`Qwen3-Reranker-0.6B`
- LLM 调用方式：OpenAI 兼容接口
- 前端：一个简单的静态测试页面

## 目录结构

```text
mini_rag_api/
+-- app/
|   +-- main.py
|   +-- rag_qdrant.py
|   +-- schemas.py
|   `-- utils.py
+-- static/
+-- storage/
+-- .env
+-- .gitignore
+-- requirements.txt
`-- README.md
```

说明：

- `app/main.py`：FastAPI 入口和接口定义
- `app/rag_qdrant.py`：RAG 主流程，包括检索、rerank、上下文构建和回答生成
- `app/schemas.py`：请求体和响应体定义
- `app/utils.py`：文档读取与文本处理工具
- `static/index.html`：本地测试页面
- `storage/`：本地 Qdrant 数据目录

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

如果你需要镜像源：

```bash
pip install -i https://mirrors.aliyun.com/pypi/simple -r requirements.txt
```

### 2. 准备本地模型

当前代码默认使用：

- `bge-small-en-v1.5` 作为 embedding 模型
- `Qwen3-Reranker-0.6B` 作为 rerank 模型

默认模型路径是：

- `/home/zhaodongmin/RAG/bge-small-en-v1.5`
- `/home/zhaodongmin/RAG/qwen3-reranker-0.6b`

如果你的模型路径不同，可以通过环境变量覆盖：

```bash
export EMBEDDING_MODEL_PATH=/你的/embedding/模型路径
export RERANK_MODEL_PATH=/你的/rerank/模型路径
```

### 3. 配置 `.env`

示例：

```env
OPENAI_API_KEY=your_api_key
OPENAI_BASE_URL=your_openai_compatible_base_url
CHAT_MODEL=your_chat_model
```

### 4. 启动服务

```bash
uvicorn app.main:app --reload
```

启动后可以打开：

- Swagger 文档：`http://localhost:8000/docs`
- 前端测试页：`http://localhost:8000/`

### 5. 建立索引

先调用：

- `POST /rag/index`

或：

- `POST /rag/index/configurable`

把 `data/` 目录中的文档切块并写入 Qdrant。

## 主要接口

- `POST /rag/chat`
- `POST /rag/index`
- `POST /rag/index/configurable`
- `POST /rag/add_text`
- `GET /documents`
- `GET /documents/{document_id}/chunks`
- `DELETE /documents/{document_id}`

## 当前版本的一些说明

- 如果更换了 embedding 模型，必须重建 Qdrant 索引
- 旧 embedding 模型生成的向量不能和新 embedding 模型生成的向量混用
- 当前测试主要基于英文问题和英文文档
- rerank 在本地执行，效果更稳定，但速度会慢于只做向量检索

## 这个仓库为什么存在

这个仓库更像一个“RAG 学习实验田”，重点不是把框架封得多漂亮，而是把链路拆开，让每个阶段都能看得见、改得动、验证得了。

我希望通过这个项目持续理解这些问题：

- 文档应该怎么切块
- embedding 检索到底在做什么
- rerank 为什么能提升结果质量
- 最终送给 LLM 的上下文应该怎么选
- 引用信息应该怎么保留和展示

## 后续可以继续做的方向

- 检索与 rerank 的评估体系
- 实验记录与参数对比
- hybrid retrieval
- 文档上传接口
- 更稳定的引用与回答格式
- 更灵活的模型切换配置

## 说明

这是一个用于学习和实验的项目。如果后面准备长期公开维护，可以再补充正式的开源许可证。
