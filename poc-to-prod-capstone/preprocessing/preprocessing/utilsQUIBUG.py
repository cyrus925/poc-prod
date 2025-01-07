import numpy as np
import pandas as pd

from tensorflow.keras.utils import Sequence
from tensorflow.keras.utils import to_categorical

from sklearn.model_selection import train_test_split


def integer_floor(float_value: float):
    """
    link to doc for numpy.floor https://numpy.org/doc/stable/reference/generated/numpy.floor.html
    """
    return int(np.floor(float_value))


class _SimpleSequence(Sequence):
    """
    Base object for fitting to a sequence of data, such as a dataset.
    link to doc : https://www.tensorflow.org/api_docs/python/tf/keras/utils/Sequence
    """

    def __init__(self, get_batch_method, num_batches_method):
        self.get_batch_method = get_batch_method
        self.num_batches_method = num_batches_method

    def __len__(self):
        return self.num_batches_method()

    def __getitem__(self, idx):
        return self.get_batch_method()


class BaseTextCategorizationDataset:
    """
    Generic class for text categorization
    data sequence generation
    """

    def __init__(self, batch_size, train_ratio=0.8):
        assert train_ratio < 1.0
        self.train_ratio = train_ratio
        self.batch_size = batch_size

    def _get_label_list(self):
        """
        returns list of labels
        should not be implemented in this class (we can assume its a given)
        """
        raise NotImplementedError

    def get_num_labels(self):
        """
        returns the number of labels
        """
        return len(self.labels)


    def _get_num_samples(self):
        """
        returns number of samples (dataset size)
        should not be implemented in this class (we can assume its a given)
        """
        raise NotImplementedError

    def _get_num_train_samples(self):
        """
        returns number of train samples
        (training set size)
        """
        num_samples = self._get_num_samples()  
        return int(np.floor(num_samples * self.train_ratio))

    def _get_num_test_samples(self):
        """
        returns number of test samples
        (test set size)
        """
        return self._get_num_samples() - self._get_num_train_samples()
    

    def _get_num_train_batches(self):
        """
        returns number of train batches
        """
        num_train_samples = self._get_num_train_samples()
        return integer_floor(num_train_samples / self.batch_size) 

    def _get_num_test_batches(self):
        """
        returns number of test batches
        """
        num_test_samples = self._get_num_test_samples()
        return integer_floor(num_test_samples / self.batch_size)

    def get_train_batch(self):
        """
        returns next train batch
        should not be implemented in this class (we can assume its a given)
        """
        raise NotImplementedError

    def get_test_batch(self):
        """
        returns next test batch
        should not be implemented in this class (we can assume its a given)
        """
        raise NotImplementedError

    def get_index_to_label_map(self):
        """
        from label list, returns a map index -> label
        (dictionary index: label)
        """
        labels = self._get_label_list()
        return {i: label for i, label in enumerate(labels)}

    def get_label_to_index_map(self):
        """
        from index -> label map, returns label -> index map
        (reverse the previous dictionary)
        """
        index_to_label = self.get_index_to_label_map()
        return {label: i for i, label in index_to_label.items()}

    def to_indexes(self, labels):
        """
        from a list of labels, returns a list of indexes
        """
        label_to_index = self.get_label_to_index_map()
        return [label_to_index[label] for label in labels]

    def get_train_sequence(self):
        """
        returns a train sequence of type _SimpleSequence
        """
        return _SimpleSequence(self.get_train_batch, self._get_num_train_batches)

    def get_test_sequence(self):
        """
        returns a test sequence of type _SimpleSequence
        """
        return _SimpleSequence(self.get_test_batch, self._get_num_test_batches)

    def __repr__(self):
        return self.__class__.__name__ + \
               f"(n_train_samples: {self._get_num_train_samples()}, " \
               f"n_test_samples: {self._get_num_test_samples()}, " \
               f"n_labels: {self.get_num_labels()})"


class LocalTextCategorizationDataset(BaseTextCategorizationDataset):
    """
    A TextCategorizationDataset read from a file residing in the local filesystem
    """

    def __init__(self, filename, batch_size,
                 train_ratio=0.8, min_samples_per_label=100, preprocess_text=lambda x: x):
        """
        :param filename: a CSV file containing the text samples in the format
            (post_id 	tag_name 	tag_id 	tag_position 	title)
        :param batch_size: number of samples per batch
        :param train_ratio: ratio of samples dedicated to training set between (0, 1)
        :param preprocess_text: function taking an array of text and returning a numpy array, default identity
        """
        super().__init__(batch_size, train_ratio)
        self.filename = filename
        self.preprocess_text = preprocess_text

        self._dataset = self.load_dataset(filename, min_samples_per_label)

        assert self._get_num_train_batches() > 0
        assert self._get_num_test_batches() > 0

        # TODO: CODE HERE
        # from self._dataset, compute the label list
        self._label_list = self._dataset['tag_name'].unique().tolist()

        

        y = self.to_indexes(self._dataset['tag_name'])
        y = to_categorical(y, num_classes=len(self._label_list))

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self._dataset['title'],
            y,
            train_size=self._get_num_train_samples(),
            stratify=y)

        self.train_batch_index = 0
        self.test_batch_index = 0
        
    @staticmethod
    def load_dataset(filename, min_samples_per_label, position=0):
        """
        Loads dataset from filename and applies pre-processing steps:
        - Keeps only rows where tag_position equals the given position.
        - Removes tags that are seen less than `min_samples_per_label` times.
        """

        # Lire le fichier CSV
        df = pd.read_csv(filename)

        # Vérifier que les colonnes nécessaires sont présentes
        required_columns = ['post_id', 'tag_name', 'tag_id', 'tag_position', 'title']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Column '{col}' is missing from the dataset")

        def filter_tag_position(df, position):
            """Garder uniquement les lignes où tag_position == position"""
            return df[df['tag_position'] == position]

        def filter_tags_with_less_than_x_samples(df):
            """Supprimer les tags avec moins de `min_samples_per_label` occurrences"""
            tag_counts = df['tag_name'].value_counts()
            valid_tags = tag_counts[tag_counts >= min_samples_per_label].index
            return df[df['tag_name'].isin(valid_tags)]

        # Appliquer les transformations via pipe
        dataset = (
            df
            .pipe(filter_tag_position, position)
            .pipe(filter_tags_with_less_than_x_samples)
        )

        return dataset
 


    # we need to implement the methods that are not implemented in the super class BaseTextCategorizationDataset

    def _get_label_list(self):
        """
        returns label list
        """
        return self._label_list

    def _get_num_samples(self):
        """
        returns number of samples in dataset
        """
        # TODO: CODE HERE
        return len(self._dataset)

    def get_train_batch(self):
        if self._get_num_train_batches() == 0:
            raise AssertionError("No training batches available. Ensure there is training data and batch size is valid.")

        i = self.train_batch_index
        start_idx = i * self.batch_size
        end_idx = (i + 1) * self.batch_size

        # Récupérer et prétraiter les données d'entrée (x_train)
        next_x = self.preprocess_text(self.x_train[start_idx:end_idx])
        
        # Récupérer les labels correspondants (y_train)
        next_y = self.y_train[start_idx:end_idx]

        # Mettre à jour l'index du batch (circulaire)
        self.train_batch_index = (self.train_batch_index + 1) % self._get_num_train_batches()

        return next_x, next_y

    def get_test_batch(self):
        """
        it does the same as get_train_batch for the test set
        """
        i = self.test_batch_index
        start_idx = i * self.batch_size
        end_idx = start_idx + self.batch_size

        # Fetch batch and apply preprocessing
        next_x = self.preprocess_text(self.x_test[start_idx:end_idx])
        next_y = self.y_test[start_idx:end_idx]

        # Update batch index
        self.test_batch_index = (self.test_batch_index + 1) % self._get_num_test_batches()

        return next_x, next_y
