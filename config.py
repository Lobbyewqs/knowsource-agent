"""
配置文件
"""
import os
import json
import hashlib
import re
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / '.env')

# 全局配置
BASE_DIR = Path(__file__).parent
DEEPSEEK_API_KEY = os.environ.get('DEEPSEEK_API_KEY')
DEEPSEEK_BASE_URL = os.environ.get('DEEPSEEK_BASE_URL') or 'https://api.deepseek.com'
DEEPSEEK_MODEL = os.environ.get('DEEPSEEK_MODEL') or 'deepseek-chat'
EMBEDDING_MODEL = os.environ.get('EMBEDDING_MODEL') or 'BAAI/bge-small-zh-v1.5'

# RAG超参数

CHUNK_SIZE = 500
CHUNK_OVERLAP = 80

VECTOR_TOP_K = 8
B25_TOP_K = 8
FINAL_TOP_K = 4

RRF_WEIGHT = 0.8
KEYWORD_WEIGHT=0.2

# 路径函数
def get_path(kb_name):
    if not re.fullmatch(r'[\w-]+', kb_name) or kb_name in {'.', '..'}:
        raise ValueError('知识库名称只能包含文字、数字、下划线和短横线')
    # 知识库路径（匹配你的目录结构 knowledge_bases/）
    knowledge_dir = BASE_DIR / 'knowledge_bases' / kb_name

    # 索引持久化（匹配 data/index/）
    index_dir = BASE_DIR / 'data' / 'index' / kb_name

    root = BASE_DIR / 'data' / kb_name
    active = root / 'active.json'
    model = EMBEDDING_MODEL
    if active.exists():
        manifest = json.loads(active.read_text(encoding='utf-8'))
        generation = manifest['generation']
        if not re.fullmatch(r'[a-f0-9]{32}', generation):
            raise ValueError('索引版本标识无效，请重建索引')
        root = root / 'versions' / generation
        model = manifest['embedding_model']
    return {
        'knowledge_dir': knowledge_dir,
        'index_dir': index_dir,
        'chroma_dir': root / 'chroma',
        'bm25_dir': root / 'chunks.json',
        'embedding_model': model,
    }
# 向量库集合哈希
def get_collection_name(kb_name):
    name_hash = hashlib.md5(kb_name.encode('utf-8')).hexdigest()[:12]
    return f'kb_{name_hash}'
