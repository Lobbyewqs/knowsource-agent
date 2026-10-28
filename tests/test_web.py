import http.client
import json
import threading
import unittest
from http.server import ThreadingHTTPServer
from unittest.mock import patch

from web_app import Handler


class WebTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        cls.port = cls.server.server_port
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def request(self, method, path, payload=None, origin=True):
        conn = http.client.HTTPConnection('127.0.0.1', self.port)
        headers = {'Content-Type': 'application/json'}
        if origin:
            headers['Origin'] = f'http://127.0.0.1:{self.port}'
        conn.request(method, path, json.dumps(payload).encode() if payload is not None else None, headers)
        response = conn.getresponse()
        data = response.read()
        conn.close()
        return response.status, data

    def test_page_and_libraries(self):
        status, body = self.request('GET', '/')
        self.assertEqual(status, 200)
        self.assertIn('今天想了解什么'.encode(), body)
        status, body = self.request('GET', '/api/libraries')
        self.assertEqual(status, 200)
        self.assertIn('demo', [x['name'] for x in json.loads(body)['libraries']])

    def test_chat_passes_history_and_returns_sources(self):
        result = {'answer': '回答 [1]', 'sources': [{'source':'guide.md', 'page':1, 'text':'原文'}]}
        with patch('web_app.answer_question', return_value=result) as answer:
            status, body = self.request('POST', '/api/chat', {'kb':'demo', 'question':'  问题  ', 'history':['上一轮']})
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body), result)
        answer.assert_called_once_with('demo', '问题', ['上一轮'])

    def test_rejects_invalid_requests(self):
        with patch('web_app.answer_question') as answer:
            for payload, origin, expected in [({'kb':'demo','question':'x'}, False, 403), ({'kb':'../hr','question':'x'}, True, 400), ({'kb':'demo','question':'  '}, True, 400)]:
                self.assertEqual(self.request('POST', '/api/chat', payload, origin)[0], expected)
            answer.assert_not_called()


if __name__ == '__main__':
    unittest.main()
