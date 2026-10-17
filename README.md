# 知源 KnowSource · 专业知识库问答 Agent

基于 DeepSeek 与混合检索的中文知识库问答助手。将 Markdown、TXT、PDF 和 Word 文档转为可检索的知识，通过多轮查询改写、向量召回、BM25 和来源引用，辅助查阅专业资料。

> 课程实践项目，当前提供命令行交互。属于固定工作流的 RAG 助手，不包含自主规划、外部工具调用或网页界面。

## 功能与流程

```text
文档 → 清洗 → 切片 → 向量索引 + BM25
问题 + 最近三轮对话 → 查询改写 → 双路召回 → RRF 融合
→ 关键词加权排序 → DeepSeek 基于证据回答 → 显示引用来源
```

- 多知识库隔离：内置 `demo` 与 `equipment` 示例文档，可自行添加专业资料。
- 中文检索：Jieba 分词、本地 Hugging Face 嵌入、Chroma 持久化。
- 可追溯回答：输出文件相对路径和页码；非 PDF 页码仅作占位，不代表原文分页。
- 索引按版本生成，构建成功才切换，失败不会删除旧索引。
- JSON 保存文本切片，运行时重建 BM25，避免加载 pickle。
- 索引记录嵌入模型，避免配置改变后使用不同模型检索旧向量。

## 快速开始

建议 Python 3.11，首次安装与模型下载需要网络和足够磁盘空间。

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
# 编辑 .env，填写自己的 DEEPSEEK_API_KEY
python build_index.py demo
python chat.py demo
```

设备库：`python chat.py equipment`。输入 `q`、`quit` 或 `exit` 结束。

示例问题：演示库可询问资料更新流程；设备库可询问“E101 报警如何处理？”并追问“再次出现怎么办？”。具体回答以本地文档为准。

## 配置与扩展

在 `knowledge_bases/<库名>/` 放入文档，再执行 `python build_index.py <库名>`。修改文档后需要重建；库名支持文字、数字、下划线及短横线。扫描型 PDF 需要先做 OCR，当前不支持直接识别扫描页。

`.env` 支持 `DEEPSEEK_API_KEY`、`DEEPSEEK_BASE_URL`、`DEEPSEEK_MODEL` 和 `EMBEDDING_MODEL`。切片、召回数和排序权重位于 `config.py`。密钥和生成索引不提交到 GitHub。

旧版索引会自动触发重建。新索引保存在 `data/<库名>/versions/`；旧版本保留以避免构建失败造成数据丢失，长期使用可在程序退出后自行清理不再使用的版本。

## 验证

```powershell
python -m unittest discover -s tests -v
python -m compileall -q config.py utils.py build_index.py chat.py docu.py
```

单元测试覆盖清洗、名称校验、索引版本、查询改写回退、融合排序和无证据回答；测试不调用在线 API。在线回答质量依赖文档、模型及服务状态，不承诺完全消除幻觉。当前排序仍使用关键词重合度过滤，可能漏掉同义表述。

## 文件说明

| 文件 | 用途 |
|---|---|
| `build_index.py` | 文档导入、切片、版本化索引 |
| `chat.py` | 多轮改写、混合检索、重排、引用问答 |
| `config.py` | 环境配置与知识库路径 |
| `utils.py` | 中文文本清洗和分词 |
| `docu.py` | 列出本地可用知识库 |
| `tests/` | 离线回归测试 |

原始 `hr` 资料含内部文件标记，保留在本地，不随仓库上传。公开示例为演示资料。

文档内容仅供课程演示；设备维护须遵循真实设备厂商说明及现场规程。提问、最近三轮对话（用于改写）和召回的文档片段会发送给所配置的模型服务，请仅使用有权处理的资料。
