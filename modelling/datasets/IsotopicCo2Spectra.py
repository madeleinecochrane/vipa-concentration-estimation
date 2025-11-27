import numpy as np
from .Spectroscopy import Spectroscopy

class IsotopicCo2Spectra(Spectroscopy):
    def __init__(self, root="processed", train=True, transform=None, config=None):
        """
        Create a new instance of the Isotopic Co2 Spectra class
        """
        self.total_classes = 2
        self.num_total_samples = 3000
        self.max_data_length = 12000

        # ratios of 12c16o2 and 13c16o2 in their natural abundance
        self.carbon_12_ratio = 0.9842
        self.carbon_13_ratio = 0.01106

        self.num_noisy_spectra_per_spectrum = 1

        super(IsotopicCo2Spectra, self).__init__(root=root, train=train, transform=transform, config=config,
                                                 num_total_samples=self.num_total_samples, transformer_encoding_length=12000)

    def _extract(self):
        """
        Extracts spectra from file and saves spectra and labels to train/validation/test pkl files.
        .pkl files contain a list of spectra (12000 x 1 numpy arrays) and labels (4 x 1 numpy array with 4 outputs -
        carbon 12 concentration, carbon 13 concentration, 0/1 indicating presence carbon 12 isotopologue,
        0/1 indicating presence of carbon 13 isotopologue)
        """
        concentration_levels = [0, 10, 25, 50, 75, 100]
        train_data = []
        train_labels = []
        val_data = []
        val_labels = []
        test_data = []
        test_labels = []

        train_test_split_file_name = self.root + ("/concentration_estimation_train_test_split.txt")
        with open(train_test_split_file_name, "r") as train_test_split_file:
            train_test_split_txt = train_test_split_file.read().splitlines()
        id2train = np.genfromtxt(train_test_split_txt, dtype=int)

        for i in range(self.num_total_samples):
            extra_samples = []
            spectra_file_name = self.root + "/images/isotopic-co2-" + str(i) + ".csv"
            with open(spectra_file_name, newline='') as f:
                spectral_txt = f.read().splitlines()
            spectral_data = np.genfromtxt(spectral_txt, dtype=np.float64)
            if self.config["num_translations"]:
                for translation in range(1, self.config["num_translations"] + 1):
                    if id2train[i] == 1:
                        start_token = -self.max_data_length - translation
                        end_token = len(spectral_data) - translation
                        sliced_spectral_data = spectral_data[start_token:end_token]
                        extra_samples.append(sliced_spectral_data)
            if self.config["noise_factor"]:
                if id2train[i] == 1:
                    for _ in range(self.num_noisy_spectra_per_spectrum):
                        extra_samples.append(self._add_noise_to_spectrum(spectral_data)[-self.max_data_length:])
            spectral_data = spectral_data[-self.max_data_length:]  # take only the last N elements

            concentration_index = i // 500
            total_co2_concentration = concentration_levels[concentration_index]
            carbon_12_concentration = total_co2_concentration * self.carbon_12_ratio
            carbon_13_concentration = total_co2_concentration * self.carbon_13_ratio
            if total_co2_concentration > 0:
                label = np.array([carbon_12_concentration, carbon_13_concentration, 1, 1])
            else:
                label = np.array([carbon_12_concentration, carbon_13_concentration, 0, 0])
            #data.append(spectral_data)

            # label is stored in two parts - first n values are 0 or 1 indicating the presence of a compound
            # next n values are between 0 and 1 indicating concentration levels. n is the total number of classes
            if len(self.config['binary_cls_tasks']):
                label_binary_cls_tasks = np.array(label[self.total_classes:])
            else:
                label_binary_cls_tasks = np.empty(shape=[0])

            if len(self.config['regression_tasks']):
                label_regression_tasks = np.array(label[:self.total_classes])
            else:
                label_regression_tasks = np.empty(shape=[0])
                # the order is always regression, binary, and then multi-class classification
            label_out = np.concatenate([label_regression_tasks, label_binary_cls_tasks], axis=0)

            if id2train[i] == 1:
                train_data.append(spectral_data)
                train_labels.append(label_out)
                if len(extra_samples):
                    for sample in extra_samples:
                        train_data.append(sample)
                        train_labels.append(label_out)
            elif id2train[i] == 2:
                val_data.append(spectral_data)
                val_labels.append(label_out)
            elif id2train[i] == 3:
                test_data.append(spectral_data)
                test_labels.append(label_out)

        self._save_train_test_data(train_data, train_labels, test_data, test_labels, val_data, val_labels)


    def _add_noise_to_spectrum(self, original_spectrum):
        """
        Add random noise to each point of a spectrum for the noisy spectra data augmentation
        original_spectrum: the original spectrum to be augmented
        @returns the augmented spectrum with gaussian noise added
        """
        gaussian_noise = np.random.normal(loc=0, scale=self.config["noise_factor"], size=len(original_spectrum))
        noisy_spectrum = original_spectrum + gaussian_noise
        return noisy_spectrum