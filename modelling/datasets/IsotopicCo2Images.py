import numpy as np
from torchvision import transforms
from .VipaImages import VipaImages

class IsotopicCo2Images(VipaImages):
    """
    Creates an instance of the Isotopic Co2 Images class.
    """
    def __init__(self, root="processed", train=True, transform=None, config=None):
        self.num_total_images = 3000
        # normalization uses the mean and standard deviation across all values in the image
        self.normalization = transforms.Normalize(mean=(0.98, 0.98, 0.98), std=(0.0761, 0.0761, 0.0761))
        self.total_classes = 2

        # ratios of 12c16o2 and 13c16o2 in their natural abundance
        self.carbon_12_ratio = 0.9842
        self.carbon_13_ratio = 0.01106

        super(IsotopicCo2Images, self).__init__(root=root, train=train, transform=transform, config=config,
                                                normalization=self.normalization,
                                                num_total_samples=self.num_total_images)

    def _extract(self):
        """
        Extracts images from file and saves them with labels in train, validation and test pkl files.
        The .pkl files consist of lists of images (3 x 320 x 256 numpy arrays) and labels
        (4 x 1 numpy array with 4 outputs - carbon 12 concentration, carbon 13 concentration, 0/1 indicating presence
        carbon 12 isotopologue, 0/1 indicating presence of carbon 13 isotopologue)
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

        for i in range(self.num_total_images):
            image = self._construct_image(i)

            concentration_index = (i-1) // 500
            total_co2_concentration = concentration_levels[concentration_index]
            carbon_12_concentration = total_co2_concentration * self.carbon_12_ratio
            carbon_13_concentration = total_co2_concentration * self.carbon_13_ratio
            if total_co2_concentration > 0:
                label = np.array([carbon_12_concentration, carbon_13_concentration, 1, 1])
            else:
                label = np.array([carbon_12_concentration, carbon_13_concentration, 0, 0])
            if id2train[i] == 1:
                train_data.append(image)
                train_labels.append(label)
            elif id2train[i] == 2:
                val_data.append(image)
                val_labels.append(label)
            elif id2train[i] == 3:
                test_data.append(image)
                test_labels.append(label)

        self._save_train_test_data(train_data, train_labels, test_data, test_labels, val_data, val_labels)