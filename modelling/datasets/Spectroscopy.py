import os
import torch
import pickle
import random
import numpy as np
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


class Spectroscopy(Dataset):
    def __init__(self, root="processed", train=True, transform=None, config=None, normalization=None,
                 num_total_samples=0, transformer_encoding_length=10000):
        """
        Creates a new instance of the base spectroscopy dataset class.
        Note that this is an abstract base class and should not be instantiated in isolation.
        root - the root directory where the data files are stored
        train - a boolean that is true if this dataset is to be used for training, false otherwise
        transform - a transform of the type torchvisions.transform with any required transformations that need to be
        applied to the data
        config - a dictionary containing configuration settings for this run
        normalization - a normalization object of the type transforms.Normalize to be applied to the data
        num_total_samples - the number of samples in all datasets (i.e. train/validation/test)
        transformer_encoding_length - the maximum length of a spectra to be used with transformers
        """
        super(Spectroscopy, self).__init__()

        self.root = root
        self.train = train
        self.transform = transform
        self.config = config
        self.normalization = normalization
        self.num_total_samples = num_total_samples
        self.transformer_encoding_length = transformer_encoding_length
        self.max_data_length = self.transformer_encoding_length

        if self._check_processed():
            if self.config["verbose"]:
                print('Train file has been extracted' if self.train else 'Test file has been extracted')
        else:
            self._extract()

        if self.train:
            self.train_data, self.train_label = pickle.load(
                open(os.path.join(self.root, 'processed/train.pkl'), 'rb')
            )
            self.train_data, self.train_label = shuffle(self.train_data, self.train_label)
        else:
            self.test_data, self.test_label = pickle.load(
                open(os.path.join(self.root, 'processed/test.pkl'), 'rb')
            )
            self.test_data, self.test_label = shuffle(self.test_data, self.test_label)

    def __len__(self):
        """
        Returns the number of samples in the dataset.
        """
        return len(self.train_data) if self.train else len(self.test_data)

    def _check_processed(self):
        """
        Check whether there are valid train and test files that already exist.
        @returns true if both <self.root>/processed/train.pkl and <self.root>/processed/test.pkl exist, false otherwise
        """
        assert os.path.isdir(self.root)
        return (os.path.isfile(os.path.join(self.root, 'processed/train.pkl')) and
                os.path.isfile(os.path.join(self.root, 'processed/test.pkl')))

    def _extract(self):
        pass

    def __getitem__(self, idx):
        """
        Gets the datapoint and label with index idx
        idx - the index of the desired item
        @returns a tuple with the original datapoint, the datapoint with any transforms applied,
        the label and an empty numpy array
        """
        if self.train:
            data, label = self.train_data[idx], self.train_label[idx]
        else:
            data, label = self.test_data[idx], self.test_label[idx]
        source_data = data
        if self.config['network'] == "transformer":
            data = data[-self.max_data_length:]
            data = np.reshape(data, (len(data.flatten()) // self.config['num_transformer_bins'],
                                     self.config['num_transformer_bins']))
            label = torch.flatten(torch.from_numpy(label))
        if self.transform is not None:
            data = self.transform(data)
        else:
            if isinstance(data, np.ndarray):
                data = torch.from_numpy(data).float()
            else:
                data = torch.from_numpy(np.array(data)).float()
        source_data = torch.from_numpy(np.array(source_data)).float()
        if self.normalization is not None:
            data = self.normalization(data)

        data = data.float()

        return source_data, data, label

    def _save_train_test_data(self, train_data, train_labels, test_data, test_labels, val_data=None, val_labels=None, train_alt_labels=None, test_alt_labels=None, val_alt_labels=None):
        """
        Save the training sets to .pkl files
        train_data - the training data saved as a list of numpy arrays
        train_labels - the training labels, saved as a list of numpy arrays
        test_data - the test data saved as a list of numpy arrays
        test_labels - the test labels, saved as a list of numpy arrays
        val_data - the validation data saved as a list of numpy arrays
        val_labels - the validation labels, saved as a list of numpy arrays
        train_alt_labels (optional) - additional metadata to be saved with the training set
        validation_alt_labels (optional) - additional metadata to be saved with the test set
        val_alt_labels (optional) - additional metadata to be saved with the validation set
        """
        if self.config["verbose"]:
            print('Total images: {}, training images: {}. testing images: {}'.format(len(train_data) + len(test_data),
                                                                                 len(train_data), len(test_data)))
        pickle.dump((train_data, train_labels),
                    open(os.path.join(self.root, 'processed/train.pkl'), 'wb'))
        pickle.dump((val_data, val_labels),
                    open(os.path.join(self.root, 'processed/test.pkl'), 'wb'))

        if test_data is not None:
            pickle.dump((test_data, test_labels),
                        open(os.path.join(self.root, 'processed/held-out-test.pkl'), 'wb'))

        if train_alt_labels is not None:
            pickle.dump((train_alt_labels, test_alt_labels, val_alt_labels),
                    open(os.path.join(self.root, 'processed/alt_labels.pkl'), 'wb'))
