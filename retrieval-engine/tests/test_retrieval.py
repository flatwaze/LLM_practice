import unittest

import numpy as np

from src.retrieval import ExactRetriever


class FakeEncoder:
    def encode(self, texts):
        vectors = {
            'a': [0, 1],
            'b': [1, 0],
            'c': [0.6, 0.8],
            'zero': [0, 0],
            'query': [1, 0],
        }
        return np.array([vectors[text] for text in texts], dtype=np.float32)


class TestRetrieval(unittest.TestCase):
    def setUp(self):
        self.retriever = ExactRetriever(
            FakeEncoder(),
            [{'document_id': 'A', 'text': 'a b'},
             {'document_id': 'B', 'text': 'c'},
             {'document_id': 'empty', 'text': ''}],
            chunk_size=1,
            overlap=0,
        )

    def test_search_chunks(self):
        results = self.retriever.search_chunks('query', top_k=2)
        self.assertEqual([(r.document_id, r.chunk_id, r.text) for r in results],
                         [('A', 1, 'b'), ('B', 0, 'c')])
        np.testing.assert_allclose([r.score for r in results], [1, 0.6])

    def test_search_documents(self):
        results = self.retriever.search_documents('query')
        self.assertEqual([(r.document_id, r.chunk_id) for r in results],
                         [('A', 1), ('B', 0)])
        np.testing.assert_allclose([r.score for r in results], [1, 0.6])
        self.assertEqual(self.retriever.search_docs('query'), results)

    def test_large_top_k(self):
        self.assertEqual(len(self.retriever.search_chunks('query', 100)), 3)
        self.assertEqual(len(self.retriever.search_documents('query', 100)), 2)

    def test_empty_index(self):
        for documents in [[], [{'document_id': 'empty', 'text': '  '}]]:
            self.retriever.index(documents)
            self.assertEqual(self.retriever.search_chunks('query'), [])
            self.assertEqual(self.retriever.search_documents('query'), [])

    def test_invalid_top_k(self):
        for method in [self.retriever.search_chunks, self.retriever.search_documents]:
            for top_k in [0, -1]:
                with self.subTest(method=method.__name__, top_k=top_k):
                    with self.assertRaises(ValueError):
                        method('query', top_k)

    def test_reindex_and_zero_vectors(self):
        self.retriever.index([{'id': 'Z', 'text': 'zero'}])
        for method in [self.retriever.search_chunks, self.retriever.search_documents]:
            for query in ['query', 'zero']:
                results = method(query)
                self.assertEqual(len(results), 1)
                self.assertEqual(results[0].document_id, 'Z')
                self.assertEqual(results[0].score, 0)

    def test_duplicate_document_ids(self):
        expected = self.retriever.search_documents('query')
        with self.assertRaises(ValueError):
            self.retriever.index([{'id': 'A', 'text': 'a'}, {'id': 'A', 'text': 'b'}])
        self.assertEqual(self.retriever.search_documents('query'), expected)

    def test_failed_encoding_preserves_index(self):
        expected = self.retriever.search_documents('query')
        with self.assertRaises(KeyError):
            self.retriever.index([{'id': 'unknown', 'text': 'unknown'}])
        self.assertEqual(self.retriever.search_documents('query'), expected)

    def test_invalid_chunking_parameters(self):
        with self.assertRaises(ValueError):
            ExactRetriever(FakeEncoder(), chunk_size=1, overlap=1)
