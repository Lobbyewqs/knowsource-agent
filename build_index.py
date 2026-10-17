"""
加载 清洗 切片 向量索引 倒排
"""
import json
import uuid
import sys
from pathlib import Path
from langchain_chroma import Chroma
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import *
from utils import clean_text, tokenize
from rank_bm25 import BM25Okapi


def load_document(knowledge_dir):
    all_docs = []
    for file_path in sorted(knowledge_dir.rglob('*')):
        if not file_path.is_file():
            continue
        suffix = file_path.suffix.lower()
        print(f"正在处理: {file_path.name}")

        try:
            if suffix in [".txt", ".md"]:
                loader_docs = TextLoader(
                    str(file_path),
                    encoding="utf-8",
                    autodetect_encoding=True
                ).load()
            elif suffix == '.pdf':
                loader_docs = PyPDFLoader(str(file_path)).load()
            elif suffix == '.docx':
                loader_docs = Docx2txtLoader(str(file_path)).load()
            else:
                print(f"    跳过不支持的文件格式: {suffix}")
                continue

            for doc in loader_docs:
                text = clean_text(doc.page_content)
                if len(text) < 20:
                    continue
                all_docs.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": file_path.relative_to(knowledge_dir).as_posix(),
                            "page": doc.metadata.get("page", 0),
                        }
                    )
                )
            print(f"    加载成功")
        except Exception as e:
            print(f"    加载失败: {e}")
            continue

    return all_docs


def build_knowledge_base(kb_name):
    # 获取路径
    paths = get_path(kb_name)
    knowledge_dir = paths['knowledge_dir']
    index_dir = paths['index_dir']
    chroma_dir = paths['chroma_dir']
    bm25_path = paths['bm25_dir']

    print(f" 知识库路径: {knowledge_dir}")

    if not knowledge_dir.exists():
        raise FileNotFoundError(f"找不到知识库目录: {knowledge_dir}")

    # 每次构建写入独立版本，全部成功后才切换当前索引。
    generation = uuid.uuid4().hex
    version_dir = BASE_DIR / 'data' / kb_name / 'versions' / generation
    chroma_dir = version_dir / 'chroma'
    bm25_path = version_dir / 'chunks.json'
    version_dir.mkdir(parents=True)

    # 加载并清洗原始文档
    documents = load_document(knowledge_dir)
    if not documents:
        raise ValueError("没有找到可用的文档")

    print(f" 加载了 {len(documents)} 篇文档")

    # 文本分块
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n## ", "\n### ", "\n\n", "\n", "？", "；", "，", " ", ""]
    )
    chunks = splitter.split_documents(documents)

    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = str(index)

    print(f" 切分成 {len(chunks)} 个文本块")

    # 构建向量语义索引
    print(" 正在构建向量索引...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    #  修复：使用新的 API 方式创建 Chroma
    # 方法1：使用 from_documents（推荐）
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,  #  改为 embedding（单数）
        collection_name=get_collection_name(kb_name),
        persist_directory=str(chroma_dir),
    )

    print(f" 向量索引已保存到: {chroma_dir}")

    # 构建BM25关键词索引
    print(" 正在构建BM25索引...")
    chunk_data = []
    tokenized_corpus = []
    for chunk in chunks:
        chunk_data.append({
            "chunk_id": chunk.metadata["chunk_id"],
            "text": chunk.page_content,
            "metadata": chunk.metadata,
        })
        tokenized_corpus.append(tokenize(chunk.page_content))

    bm25 = BM25Okapi(tokenized_corpus)

    bm25_path.write_text(json.dumps(chunk_data, ensure_ascii=False), encoding='utf-8')
    active = BASE_DIR / 'data' / kb_name / 'active.json'
    temporary = active.with_name(f'active-{generation}.tmp')
    temporary.write_text(json.dumps({'generation': generation, 'embedding_model': EMBEDDING_MODEL}), encoding='utf-8')
    temporary.replace(active)
    print(f" BM25索引已保存到: {bm25_path}")

    return len(documents), len(chunks)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("用法: python build_index.py <kb_name>")
        sys.exit(1)

    try:
        docs, chunks = build_knowledge_base(sys.argv[1])
        print(f"\n 知识库: {sys.argv[1]} 构建完成!")
        print(f"   - 文档数: {docs}")
        print(f"   - 切片数: {chunks}")
    except Exception as e:
        print(f"\n 构建失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)