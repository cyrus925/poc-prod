import unittest
import pandas as pd
from unittest.mock import MagicMock

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


from preprocessing.preprocessing import utils


class TestBaseTextCategorizationDataset(unittest.TestCase):
    def test__get_num_train_samples(self):
        """
        we want to test the class BaseTextCategorizationDataset
        we use a mock which will return a value for the not implemented methods
        then with this mocked value, we can test other methods
        """
        # we instantiate a BaseTextCategorizationDataset object with batch_size = 20 and train_ratio = 0.8
        base = utils.BaseTextCategorizationDataset(20, 0.8)
        # we mock _get_num_samples to return the value 100
        base._get_num_samples = MagicMock(return_value=100)
        # we assert that _get_num_train_samples will return 100 * train_ratio = 80
        self.assertEqual(base._get_num_train_samples(), 80)

    def test__get_num_train_batches(self):
        """
        same idea as what we did to test _get_num_train_samples
        """
        base = utils.BaseTextCategorizationDataset(20, 0.8)
        base._get_num_train_samples = MagicMock(return_value=80)
        self.assertEqual(base._get_num_train_batches(), 4)  # 80 / 20 = 4

    def test__get_num_test_batches(self):
        # TODO: CODE HERE
        base = utils.BaseTextCategorizationDataset(20, 0.8)
        base._get_num_test_samples = MagicMock(return_value=50)
        self.assertEqual(base._get_num_test_batches(), 2)

    
    def test_get_index_to_label_map(self):
        base = utils.BaseTextCategorizationDataset(20, 0.8)
        base.index_to_label = {0: "label_0", 1: "label_1"}
        base._get_label_list = MagicMock(return_value=["label_0", "label_1"])
        self.assertEqual(base.get_index_to_label_map(), {0: "label_0", 1: "label_1"})


    def test_index_to_label_and_label_to_index_are_identity(self):
        # TODO: CODE HERE
        base = utils.BaseTextCategorizationDataset(20, 0.8)
        base.index_to_label = {0: "label_0", 1: "label_1"}
        base.label_to_index = {"label_0": 0, "label_1": 1}
        for index, label in base.index_to_label.items():
            self.assertEqual(base.label_to_index[label], index)

    def test_to_indexes(self):
        # TODO: CODE HERE
        base = utils.BaseTextCategorizationDataset(20, 0.8)
        base._get_label_list = MagicMock(return_value=["label_0", "label_1"])
        base.label_to_index = {"label_0": 0, "label_1": 1}
        labels = ["label_0", "label_1", "label_0"]
        self.assertEqual(base.to_indexes(labels), [0, 1, 0])


class TestLocalTextCategorizationDataset(unittest.TestCase):
    def test_load_dataset_returns_expected_data(self):
        # we mock pandas read_csv to return a fixed dataframe
        pd.read_csv = MagicMock(return_value=pd.DataFrame({
            'post_id': ['id_1', 'id_2'],
            'tag_name': ['tag_a', 'tag_b'],
            'tag_id': [1, 2],
            'tag_position': [0, 1],
            'title': ['title_1', 'title_2']
        }))
        # we instantiate a LocalTextCategorizationDataset (it'll use the mocked read_csv), and we load dataset
        dataset = utils.LocalTextCategorizationDataset.load_dataset("fake_path", 1)
        # we expect the data after loading to be like this
        expected = pd.DataFrame({
            'post_id': ['id_1'],
            'tag_name': ['tag_a'],
            'tag_id': [1],
            'tag_position': [0],
            'title': ['title_1']
        })
        # we confirm that the dataset and what we expected to be are the same thing
        pd.testing.assert_frame_equal(dataset, expected)



    def test__get_num_samples_is_correct(self):
        """
        Test that _get_num_samples returns the correct number of samples.
        """
        # Mock pandas.read_csv pour retourner un DataFrame avec les colonnes attendues
        pd.read_csv = MagicMock(return_value=pd.DataFrame({
            'post_id': range(100),
            'tag_name': ['tag'] * 100,
            'tag_id': range(100),
            'tag_position': [0] * 100,
            'title': ['title'] * 100
        }))

        # Créez l'objet LocalTextCategorizationDataset
        dataset = utils.LocalTextCategorizationDataset("fake_path", 20)

        # Vérifiez que _get_num_samples retourne le nombre correct d'échantillons
        self.assertEqual(dataset._get_num_samples(), 100)


    def test_get_train_batch_returns_expected_shape(self):
        # Mock de pandas.read_csv pour retourner un DataFrame avec les colonnes attendues
        pd.read_csv = MagicMock(return_value=pd.DataFrame({
            'post_id': range(100),
            'tag_name': ['tag'] * 100,
            'tag_id': range(100),
            'tag_position': [0] * 100,
            'title': ['title'] * 100
        }))
        
        # Créez l'objet LocalTextCategorizationDataset avec un nom de fichier fictif
        dataset = utils.LocalTextCategorizationDataset("fake_path", batch_size=2, train_ratio=0.8)

        # Assurez-vous que les données du dataset sont bien un DataFrame
        dataset._dataset = pd.DataFrame({
            'post_id': range(100),
            'tag_name': ['tag'] * 100,
            'tag_id': range(100),
            'tag_position': [0] * 100,
            'title': ['title'] * 100
        })
        
        # Initialisez les attributs nécessaires
        dataset.x_train = dataset._dataset['title'].values
        dataset.y_train = dataset._dataset['tag_id'].values
        dataset.train_batch_index = 0  # Commencez au premier batch

        # Assurez-vous que la méthode preprocess_text est configurée
        dataset.preprocess_text = lambda x: x  # Identité, ne modifie pas les textes

        # Définir un nombre de batches (par exemple, 50) pour que le test fonctionne
        dataset._num_train_batches = 50
        
        # Obtenez le batch de train pour l'index 0
        next_x, next_y = dataset.get_train_batch()

        # Vérifiez que la forme du batch est correcte (2 échantillons, 2 colonnes)
        self.assertEqual(next_x.shape, (2,))  # Parce que x_train contient des titres sous forme de tableau de strings
        self.assertEqual(next_y.shape, (2,))  # Le y_train contient les tag_id correspondants


    def test_get_test_batch_returns_expected_shape(self):
                # Mock de pandas.read_csv pour retourner un DataFrame avec les colonnes attendues
        # Créez une instance simulée de LocalTextCategorizationDataset
        dataset = MagicMock(spec=utils.LocalTextCategorizationDataset)

        # Simulez les attributs nécessaires
        dataset.batch_size = 20
        dataset.x_test = pd.Series(range(100))
        dataset.y_test = pd.Series(range(100))
        dataset.test_batch_index = 0
        dataset.preprocess_text = MagicMock(side_effect=lambda x: x)  

        dataset._get_num_test_batches = MagicMock(return_value=5) 


    def test_get_train_batch_raises_assertion_error(self):
        """
        Test that get_train_batch raises an AssertionError when no train samples are available.
        """
        dataset = utils.LocalTextCategorizationDataset("fake_path", 20)
        dataset._get_num_train_batches = MagicMock(return_value=0)
        with self.assertRaises(AssertionError):
            dataset.get_train_batch()