"""
Author: Madeleine Cochrane
Last Modified: 02 SEPT 2025
"""

import os
import numpy as np
from .Spectroscopy import Spectroscopy

class VipaImages(Spectroscopy):
    def __init__(self, root="processed", train=True, transform=None, config=None, normalization=None,
                 num_total_samples=0):
        """
        Creates a new instance of the base VIPA Image dataset class.
        This class is used to construct and load VIPA images from file.
        """
        super(VipaImages, self).__init__(root=root, train=train, transform=transform, config=config,
                                         normalization=normalization, num_total_samples=num_total_samples)

    def _get_image_from_file(self, image_type, image_num, dir=''):
        """
        Retrieves a specified VIPA image from a binary file.
        This method assumes that the image is stored in a filename of format:
            <self.root>/<dir>/<image_type>_<image_num>.bin
        image_type: descriptor used in filename of desired image
        image_num: image number of the desired image
        dir: path from self.root to the directory where the desired image is stored.
        @returns the image as a 320 x 256 numpy array
        """
        filepath = image_type + "_" + str(image_num) + ".bin"
        filepath = os.path.join(self.root, dir, filepath)
        with open(filepath, 'rb') as file:
            return np.fromfile(file, np.int16).reshape((320, 256)).T

    def _construct_image(self, image_num, dir=''):
        """
        Constructs a single VIPA image, ready for processing, from 4 VIPA images.
        The assumed 4 images are Sig/DarkSig/Ref/DarkRef
        image_num: image number of the desired image
        dir: path from self.root to the directory where the desired image is stored.
        @returns the image as a 3 x 320 x 256 numpy array
        """
        dark_ref = self._get_image_from_file("DarkRef", image_num + 1, dir)
        ref = self._get_image_from_file("Ref", image_num + 1, dir)
        dark_sig = self._get_image_from_file("DarkSig", image_num + 1, dir)
        sig = self._get_image_from_file("Sig", image_num + 1, dir)

        ref_corrected_image = ref - dark_ref
        sig_corrected_image = sig - dark_sig

        image = np.divide(sig_corrected_image, ref_corrected_image, out=np.zeros_like(sig_corrected_image, float),
                          where=ref_corrected_image != 0)

        # two of the images have outlier pixels, this clips them in line with other values
        image = np.clip(image, 0., 2.)

        # resnet operates on rgb images so add two additional channels that repeat the first channel
        image = np.repeat(image[:, :, np.newaxis], 3, axis=2)

        return image