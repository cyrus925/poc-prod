import unittest
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from preprocessing.preprocessing.embeddings import embed


class EmbeddingsTest(unittest.TestCase):
    def test_embed(self):
        embeddings = embed(['hello world'])
        self.assertEqual(embeddings.shape, (1, 768))
