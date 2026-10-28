"""本地 Web 服务；模型与索引仅在首次提问时加载。"""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from config import BASE_DIR, DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL, get_collection_name, get_path

WEB = BASE_DIR / 'web'
lock = threading.Lock()


def libraries():
    root = BASE_DIR / 'knowledge_bases'
    return [{'name': p.name, 'indexed': get_path(p.name)['bm25_dir'].exists()}
            for p in sorted(root.iterdir()) if p.is_dir() and not p.name.startswith('.')]


def answer_question(kb, question, history):
    from rank_bm25 import BM25Okapi
    from langchain_chroma import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_openai import ChatOpenAI
    from build_index import build_knowledge_base
    from chat import rewrite_query, hybrid_retrieve, rerank, generate_answer
    from utils import tokenize

    if not DEEPSEEK_API_KEY:
        raise ValueError('请先在 .env 中配置 DEEPSEEK_API_KEY，重启服务后再提问。')
    paths = get_path(kb)
    if not paths['knowledge_dir'].is_dir():
        raise ValueError('知识库不存在，请刷新页面。')
    if not paths['chroma_dir'].exists() or not paths['bm25_dir'].exists():
        build_knowledge_base(kb)
        paths = get_path(kb)
    embeddings = HuggingFaceEmbeddings(model_name=paths['embedding_model'], model_kwargs={'device': 'cpu'}, encode_kwargs={'normalize_embeddings': True})
    vector_db = Chroma(collection_name=get_collection_name(kb), embedding_function=embeddings, persist_directory=str(paths['chroma_dir']))
    chunks = json.loads(paths['bm25_dir'].read_text(encoding='utf-8'))
    data = {'chunks': chunks, 'bm25': BM25Okapi([tokenize(c['text']) for c in chunks])}
    llm = ChatOpenAI(model=DEEPSEEK_MODEL, api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL, temperature=0.5, timeout=60, max_retries=1)
    rewritten = rewrite_query(llm, question, history)
    sources = rerank(rewritten, hybrid_retrieve(rewritten, vector_db, data))
    answer = generate_answer(llm, rewritten, sources)
    return {'answer': answer, 'sources': [{'source': item['metadata'].get('source', '未知文档'), 'page': item['metadata'].get('page', 0) + 1, 'text': item['text']} for item in sources]}


class Handler(BaseHTTPRequestHandler):
    def send_json(self, status, data):
        body = json.dumps(data, ensure_ascii=False).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlparse(self.path).path
        if path == '/api/libraries':
            return self.send_json(200, {'libraries': libraries()})
        files = {'/': ('index.html', 'text/html'), '/style.css': ('style.css', 'text/css'), '/app.js': ('app.js', 'text/javascript')}
        if path not in files:
            self.send_error(404)
            return
        name, mime = files[path]
        body = (WEB / name).read_bytes()
        self.send_response(200)
        self.send_header('Content-Type', mime + '; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if self.path != '/api/chat':
            self.send_error(404)
            return
        origin = self.headers.get('Origin')
        if origin != f'http://{self.headers.get("Host")}':
            return self.send_json(403, {'error': '请求来源无效。'})
        try:
            size = int(self.headers.get('Content-Length', '0'))
            if not 0 < size <= 32768:
                raise ValueError('请求内容过长。')
            data = json.loads(self.rfile.read(size))
            kb, question, history = data['kb'], data['question'], data.get('history', [])
            if not isinstance(kb, str) or not isinstance(question, str) or not question.strip() or len(question) > 4000:
                raise ValueError('请输入不超过 4000 字的问题。')
            if not isinstance(history, list) or len(history) > 3 or any(not isinstance(x, str) or len(x) > 8000 for x in history):
                raise ValueError('对话历史格式不正确。')
            get_path(kb)
            with lock:
                result = answer_question(kb, question.strip(), history)
            self.send_json(200, result)
        except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            self.send_json(400, {'error': str(exc)})
        except Exception:
            self.send_json(500, {'error': '处理失败，请检查本地服务日志、模型和网络配置。'})
            import traceback
            traceback.print_exc()


if __name__ == '__main__':
    server = ThreadingHTTPServer(('127.0.0.1', 8000), Handler)
    print('知源已启动：http://127.0.0.1:8000', flush=True)
    server.serve_forever()
