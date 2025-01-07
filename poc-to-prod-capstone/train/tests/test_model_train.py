import unittest
from unittest.mock import MagicMock
import tempfile

import pandas as pd
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from train.train import run
from preprocessing.preprocessing import utils


def load_dataset_mock():
    titles = [
        "Is it possible to execute the procedure of a function in the scope of the caller?",
        "ruby on rails: how to change BG color of options in select list, ruby-on-rails",
        "Is it possible to execute the procedure of a function in the scope of the caller?",
        "ruby on rails: how to change BG color of options in select list, ruby-on-rails",
        "Is it possible to execute the procedure of a function in the scope of the caller?",
        "ruby on rails: how to change BG color of options in select list, ruby-on-rails",
        "Is it possible to execute the procedure of a function in the scope of the caller?",
        "ruby on rails: how to change BG color of options in select list, ruby-on-rails",
        "Is it possible to execute the procedure of a function in the scope of the caller?",
        "ruby on rails: how to change BG color of options in select list, ruby-on-rails",
    ]
    tags = ["php", "ruby-on-rails", "php", "ruby-on-rails", "php", "ruby-on-rails", "php", "ruby-on-rails",
            "php", "ruby-on-rails"]

    return pd.DataFrame({
        'title': titles,
        'tag_name': tags
    })


class TestTrain(unittest.TestCase):
    # Remplacer la méthode load_dataset par notre mock
    utils.LocalTextCategorizationDataset.load_dataset = MagicMock(return_value=load_dataset_mock())

    def test_train(self):
        # Créer les paramètres d'entraînement
        params = {
            'batch_size': 2,
            'epochs': 5,
            'dense_dim': 64,
            'min_samples_per_label': 2,
            'verbose': 1
        }

        # Créer un dossier temporaire et lancer l'entraînement
        with tempfile.TemporaryDirectory() as model_dir:
            accuracy, _ = run.train(
                dataset_path='fake_path',
                train_conf=params,
                model_path=model_dir,
                add_timestamp=False
            )

        # Vérifier que l'accuracy est égale à 1.0
        self.assertEqual(accuracy, 1.0)

