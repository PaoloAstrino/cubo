[🇬🇧 English](README.md) | [🇮🇹 Italiano](README.it.md) | [🇨🇳 中文](README.zh.md)

<div align="center">

# 🧊 CUBO
### 工业级本地 RAG 系统

**在消费级笔记本电脑上运行企业级文档搜索。100% 离线。符合 GDPR 标准。**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Testing: Pytest](https://img.shields.io/badge/tests-passing-green.svg)](tests/)
[![arXiv](https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg)](https://arxiv.org/abs/XXXX.XXXXX)
[![Stars](https://img.shields.io/github/stars/PaoloAstrino/CUBO?style=social)](https://github.com/PaoloAstrino/CUBO)

</div>

<!-- Demo / GIF 占位符：将 XXXXX 替换为实际 YouTube ID 或添加 assets/demo.gif -->
[![观看 90 秒演示](https://img.youtube.com/vi/XXXXX/0.jpg)](https://www.youtube.com/watch?v=XXXXX)
_90 秒演示：拖放 5 GB 合同 → 摄取 → 查询“解除条款” → 带引用的答案_  
_或_ `![demo](assets/demo.svg)`

---

**CUBO** 是一个专为**隐私优先**环境构建的检索增强生成（RAG）系统。它完全在您的本地机器上运行，可以处理数 GB 的文档，而无需向云端发送任何字节。

与简单的 RAG 包装器不同，CUBO 专为**受限硬件**（16GB RAM 笔记本电脑）和**欧洲语言**进行了工程设计。

## ✨ 为什么选择 CUBO？

| 特性 | 为什么重要 |
| :--- | :--- |
| **🚀 笔记本模式** | 智能资源管理（Float16，延迟加载）允许您在 **8GB/16GB RAM** 上运行海量索引。 |
| **🌍 欧洲核心** | 针对**意大利语、法语、德语和西班牙语**的高级分词。自动匹配 "gatto" 和 "gatti"。 |
| **🛡️ 100% 离线** | 没有 OpenAI。没有 Pinecone。没有 Weaviate。您的数据**永远不会**离开您的 SSD。 |
| **⚡ 流式传输** | 即使在纯 CPU 硬件上，实时令牌生成也感觉瞬间完成。 |
| **🧠 智能摄取** | 流式 Parquet 摄取处理 **50GB+ 语料库** 而不会导致 RAM 崩溃。 |

## 🚀 快速开始

**Windows (PowerShell):**
```powershell
.\run_local.ps1
```
*此脚本将检查 Python、Node.js 和 Ollama，下载所需的模型（~2GB），并启动应用程序。*

**手动安装:** 
```bash
pip install -r requirements.txt
python scripts/start_fullstack.py --mode laptop
```

## 下载与运行

[![最新版本](https://img.shields.io/github/v/release/PaoloAstrino/CUBO?color=green)](https://github.com/PaoloAstrino/CUBO/releases/latest)

- Windows: [CUBO.exe](https://github.com/PaoloAstrino/CUBO/releases/latest/download/CUBO.exe) (~180 MB)  
- Linux: [CUBO_linux](https://github.com/PaoloAstrino/CUBO/releases/latest/download/CUBO_linux) (PyInstaller)

## 📚 文档

面向开发人员和研究人员的详细指南：

- **[安装指南](docs/API_INTEGRATION.md)** - 完整的设置说明。
- **[架构与优化](docs/optimization/resource_architecture.md)** - 我们如何节省 50% 的 RAM。
- **[基准测试](docs/eval/evaluation_antigravity.md)** - Recall@10，nDCG 和速度统计。
- **[科学论文](paper/paper.pdf)** - CUBO 背后的学术理论。

## 🛠️ 架构

CUBO 使用**分层混合检索**策略：
1.  **摄取：** 文档被分块（结构感知）并流式传输到磁盘。
2.  **索引：** 向量被量化（Float16）并存储在 SQLite（元数据）+ FAISS（搜索）中。
3.  **检索：** **倒数排名融合 (RRF)** 结合了 BM25（关键字）和 Embedding（语义）分数。
4.  **生成：** 通过 Ollama 的本地 LLM（Llama 3，Mistral）生成带有引用的答案。

## 🧪 评估

我们相信测量，而不是猜测。
*   **Recall@10:** 0.96 (政治), 0.82 (跨领域)。
*   **延迟:** < 300ms 每次查询 (缓存)。
*   **摄取:** ~150 页/秒。

## 真实基准测试 (embedding-gemma-300m, 16 GB 笔记本电脑)

| 数据集          | 领域     | Recall@10 | 评价        |
|-------------------|------------|-----------|----------------|
| UltraDomain-Legal | 法律      | 0.48      | ⭐ 强大       |
| Politics          | 结构化 | 0.97      | 🚀 完美     |
| NFCorpus          | 医学    | 0.17      | ⚠️ 领域偏见|
| RAGBench-full     | 混合困难 | 0.30      | ⭐ 行业可接受|

_说明：在结构化法律文本上表现强劲（我们的主要用例），在专业术语上较弱（可通过路由器解决）。_ 

查看 [examples/04_evaluation_benchmarking.ipynb](examples/04_evaluation_benchmarking.ipynb) 运行您自己的基准测试。

## CUBO 适合谁？

- 意大利律师事务所，无法将案件文件上传到云端（89% 根据我们的调查）
- 医疗从业者，拥有敏感病人记录
- 独立研究人员，希望使用本地 RAG 而不需 AWS 费用
- 只有 16 GB 笔记本电脑且需要绝对隐私的人

## 🤝 贡献

欢迎贡献！请参阅 [CONTRIBUTING.md](docs/CONTRIBUTING.md) 了解我们的行为准则和开发流程的详细信息。

---

<div align="center">
  <sub>❤️ 专为隐私和效率打造。</sub>
</div>
