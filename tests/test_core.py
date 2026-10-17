import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import config
from chat import generate_answer, hybrid_retrieve, rerank, rewrite_query
from utils import clean_text, tokenize
from rank_bm25 import BM25Okapi


class KnowledgeTests(unittest.TestCase):
    def test_clean_bom_and_duplicate(self):
        self.assertEqual(clean_text('\ufeff标题\r\n标题\n  a\t  b '), '标题\na b')

    def test_tokenize(self):
        self.assertIn('e101', tokenize('E101 报警！'))

    def test_reject_path_escape(self):
        for name in ['../secret', 'a/b', '..', 'C:\\data', '']:
            with self.subTest(name=name), self.assertRaises(ValueError):
                config.get_path(name)

    def test_generation_model(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(config, 'BASE_DIR', Path(folder)):
            root = Path(folder) / 'data' / 'hr'
            root.mkdir(parents=True)
            version = 'a' * 32
            (root / 'active.json').write_text(json.dumps({'generation': version, 'embedding_model': 'test-model'}))
            paths = config.get_path('hr')
            self.assertEqual(paths['embedding_model'], 'test-model')
            self.assertEqual(paths['bm25_dir'], root / 'versions' / version / 'chunks.json')

    def test_invalid_manifest(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(config, 'BASE_DIR', Path(folder)):
            root = Path(folder) / 'data' / 'hr'
            root.mkdir(parents=True)
            (root / 'active.json').write_text(json.dumps({'generation': '../outside'}))
            with self.assertRaises(ValueError):
                config.get_path('hr')

    def test_rewrite_fallback(self):
        llm = Mock()
        llm.invoke.return_value.content = '   '
        self.assertEqual(rewrite_query(llm, '问题', []), '问题')
        llm.invoke.side_effect = RuntimeError('offline')
        self.assertEqual(rewrite_query(llm, '问题', []), '问题')

    def test_rewrite_uses_recent_history(self):
        llm = Mock()
        llm.invoke.return_value.content = ' E101 再次出现怎么办？ '
        self.assertEqual(rewrite_query(llm, '再次出现呢', ['old', 'a', 'b', 'c']), 'E101 再次出现怎么办？')
        self.assertNotIn('old', llm.invoke.call_args.args[0])

    def test_hybrid_deduplicates_and_ranks(self):
        chunks = [dict(chunk_id=str(i), text=text, metadata={'chunk_id': str(i)}) for i, text in enumerate(['E101 压力 报警', '休假 流程', '工资 标准'])]
        db = Mock()
        doc = Mock(page_content=chunks[0]['text'], metadata=chunks[0]['metadata'])
        db.similarity_search.return_value = [doc]
        data = {'chunks': chunks, 'bm25': BM25Okapi([tokenize(c['text']) for c in chunks])}
        result = hybrid_retrieve('E101', db, data)
        self.assertEqual(len([c for c in result if c['chunk_id'] == '0']), 1)
        top = rerank('E101', result)[0]
        self.assertEqual(top['chunk_id'], '0')
        self.assertTrue(top['from_vector'] and top['from_bm25'])

    def test_no_evidence_no_api(self):
        llm = Mock()
        self.assertIn('没有检索到', generate_answer(llm, '问题', []))
        llm.invoke.assert_not_called()
        self.assertEqual(rerank('问题', []), [])

    def test_answer_contains_citations(self):
        llm = Mock()
        llm.invoke.return_value.content = '答复 [1]'
        self.assertEqual(generate_answer(llm, 'E101?', [{'text': '压力低', 'metadata': {'source': 'manual.pdf', 'page': 2}}]), '答复 [1]')
        self.assertIn('[1] 来源：manual.pdf，页码：3', llm.invoke.call_args.args[0])


if __name__ == '__main__':
    unittest.main()
