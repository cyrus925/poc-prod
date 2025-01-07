import unittest
from unittest.mock import patch, MagicMock
import tempfile
import json
import os
import numpy as np
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from predict.predict.run import TextPredictionModel


class TestPredict(unittest.TestCase):

    def setUp(self):
        # Créer un dossier temporaire pour les artefacts
        self.test_dir = tempfile.mkdtemp()
        
        # Créer des fichiers d'artefacts factices
        self.model_path = os.path.join(self.test_dir, "model.h5")
        
        # Paramètres d'entraînement factices
        self.params = {
            "batch_size": 32,
            "dense_dim": 64,
            "epochs": 1,
            "min_samples_per_label": 10,
            "verbose": 1
        }
        with open(os.path.join(self.test_dir, "params.json"), "w") as f:
            json.dump(self.params, f)
        
        # Index des labels factices
        self.labels_to_index = {"php": 0, "python": 1, "javascript": 2}
        with open(os.path.join(self.test_dir, "labels_index.json"), "w") as f:
            json.dump(self.labels_to_index, f)

    @patch('predict.predict.run.load_model')
    @patch('predict.predict.run.embed')
    def test_predict(self, mock_embed, mock_load_model):
        # Configuration des mocks
        mock_model = MagicMock()
        mock_load_model.return_value = mock_model
        
        # Simuler les embeddings BERT (vecteurs de dimension 768)
        mock_embed.return_value = np.array([[0.5] * 768])
        
        # Simuler les prédictions du modèle (3 classes: php, python, javascript)
        mock_model.predict.return_value = np.array([[0.1, 0.8, 0.1]])
        
        # Initialiser le modèle
        model = TextPredictionModel.from_artefacts(self.test_dir)
        
        # Faire une prédiction
        text = ["How to use Python with Django?"]
        predictions = model.predict(text, top_k=2)
        
        # Vérifications
        self.assertIsInstance(predictions, list)
        self.assertEqual(len(predictions), 1)  # Une prédiction pour un texte
        self.assertEqual(len(predictions[0]), 2)  # top_k=2
        self.assertEqual(predictions[0][0], "python")  # Premier choix devrait être python
        
        # Vérifier que les mocks ont été appelés correctement
        mock_embed.assert_called_once_with(text)
        mock_model.predict.assert_called_once()
        mock_load_model.assert_called_once_with(os.path.join(self.test_dir, "model.h5"))

    @patch('predict.predict.run.load_model')
    def test_from_artefacts(self, mock_load_model):
        # Configuration du mock
        mock_model = MagicMock()
        mock_load_model.return_value = mock_model
        
        # Initialiser le modèle depuis les artefacts
        model = TextPredictionModel.from_artefacts(self.test_dir)
        
        # Vérifications
        self.assertEqual(model.params, self.params)
        self.assertEqual(model.labels_to_index, self.labels_to_index)
        self.assertEqual(
            model.labels_index_inv,
            {0: "php", 1: "python", 2: "javascript"}
        )
        mock_load_model.assert_called_once_with(os.path.join(self.test_dir, "model.h5"))

    def tearDown(self):
        # Nettoyer le dossier temporaire
        import shutil
        shutil.rmtree(self.test_dir) 