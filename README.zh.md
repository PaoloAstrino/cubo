# CUBO - AI 文档助手 v1.3.0

**注 (2025-12-17):** 文档清理 — 对齐版本信息，修正 `src`→`cubo` 示例，移除重复的依赖部分，并更新 pytest 覆盖标志。

🌍 **[English](README.md)** | **[Italiano](README.it.md)** | **中文**

[![CI/CD](https://github.com/your-username/cubo/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/your-username/cubo/actions/workflows/ci-cd.yml)
[![E2E Tests](https://github.com/your-username/cubo/actions/workflows/e2e.yml/badge.svg)](https://github.com/your-username/cubo/actions/workflows/e2e.yml)

一个模块化的检索增强生成（RAG）系统，使用嵌入模型和大型语言模型（LLM），配备现代桌面界面和 Web API。**隐私优先设计**：完全离线运行，您的数据永远不会离开您的计算机。

## 🌟 为什么选择 CUBO？

- **🔒 隐私优先**: 100% 离线 - 无数据发送到外部服务器
- **💻 笔记本友好**: 针对有限资源的笔记本电脑优化（≤16GB RAM）
- **🇪🇺 GDPR 合规**: 可导出审计日志、文档删除、查询脱敏
- **🚀 易于使用**: 无需技术配置

## v1.3.1 新功能

- **📝 引用 API**: 每个响应包含符合 GDPR 的引用，包括源文件、页码、chunk_id
- **🗑️ 文档删除**: DELETE 端点用于从索引中删除文档（GDPR 第17条）
- **📊 GDPR 审计导出**: 以 CSV/JSON 格式导出日志用于合规审计
- **🌍 多语言 README**: 中文、英文和意大利文文档

## 快速开始

### Web 界面（推荐）

```bash
# 安装依赖
pip install -r requirements.txt
cd frontend && pnpm install && cd ..

# 启动后端和前端
python scripts/start_fullstack.py
```

访问:
- 前端: http://localhost:3000
- API 文档: http://localhost:8000/docs

### 桌面界面

```bash
pip install -r requirements.txt
python launch_gui.py
```

## API 端点

### 带引用的查询

```bash
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"query": "合同中关于保修的规定是什么？", "top_k": 5}'
```

**响应:**
```json
{
  "answer": "合同规定保修期为24个月...",
  "sources": [...],
  "citations": [
    {
      "source_file": "contract_2024.pdf",
      "page": 5,
      "chunk_id": "abc123",
      "chunk_index": 12,
      "text_snippet": "保修期自交付之日起24个月...",
      "relevance_score": 0.95
    }
  ],
  "trace_id": "tr_abc123",
  "query_scrubbed": false
}
```

### 文档删除（GDPR 第17条）

```bash
curl -X DELETE http://localhost:8000/api/documents/contract_2024.pdf
```

**响应:**
```json
{
  "doc_id": "contract_2024.pdf",
  "deleted": true,
  "chunks_removed": 15,
  "trace_id": "tr_xyz789",
  "message": "Document contract_2024.pdf deleted successfully"
}
```

### GDPR 审计导出

```bash
# 导出审计为 CSV
curl "http://localhost:8000/api/export-audit?start_date=2024-11-01&format=csv" > audit.csv

# 导出为 JSON
curl "http://localhost:8000/api/export-audit?format=json" > audit.json
```

## 功能特性

- **🔍 混合搜索**: 结合 BM25（关键词）和 FAISS（语义）以获得精确结果
- **🐬 Dolphin 处理**: 使用视觉语言模型进行高级 PDF/图像解析
- **📊 语义去重**: 使用 MinHash + FAISS + HDBSCAN 减少重复块
- **🔄 重排序器**: 使用交叉编码器改善结果排序
- **📝 句子窗口**: 可配置上下文窗口的智能分块
- **🔒 查询脱敏**: 日志中查询的自动匿名化

## 笔记本模式

CUBO 自动检测笔记本硬件（≤16GB RAM 或 ≤6 个 CPU 核心）并启用优化：

```bash
# 强制笔记本模式
export CUBO_LAPTOP_MODE=1
python start_api_server.py --mode laptop
```

## 系统要求

- Python 3.8+
- Ollama（用于 LLM 生成）
- CUDA 兼容 GPU（可选）

### 快速安装

```bash
# 创建虚拟环境
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# 安装依赖
pip install -r requirements.txt

# 启动 Ollama 并下载模型
ollama pull llama3.2:latest
```

## 配置

编辑 `config.json`：

```json
{
  "model_path": "./models/all-MiniLM-L6-v2",
  "llm_model": "llama3.2:latest",
  "top_k": 5,
  "chunk_size": 512
}
```

### 环境变量

```bash
export CUBO_ENCRYPTION_KEY="your-key"
export CUBO_LAPTOP_MODE=1
```

## 企业用例

### 法律与合规
- 合同和条款分析
- 法规文件搜索
- 尽职调查

### 医疗与研究
- 病历查询（符合 HIPAA）
- 科学文献综述
- 临床指南

### 技术文档
- 企业知识库
- API 文档
- 故障排除

## 测试

```bash
# 运行所有测试
python -m pytest -q

# 带覆盖率的测试
python -m pytest --cov=cubo --cov-report=html
```

## 安全性

- **路径净化**: 防止目录遍历攻击
- **文件限制**: 可配置的最大尺寸
- **速率限制**: 防止滥用
- **加密**: 使用 Fernet 加密敏感数据
- **审计日志**: 记录操作以符合合规要求

## 许可证

MIT 许可证 - 见 [LICENSE](LICENSE)

## 贡献

见 [CONTRIBUTING.md](CONTRIBUTING.md) 了解指南。

## 支持

- 📖 [API 文档](docs/API_INTEGRATION.md)
- 🐛 [报告问题](https://github.com/your-username/cubo/issues)
- 💬 [讨论区](https://github.com/your-username/cubo/discussions)
