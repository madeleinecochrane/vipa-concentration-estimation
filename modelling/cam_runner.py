import os
import cv2
import torch
import torchvision
import numpy as np
from modelling.model import get_network
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision import transforms
from mpl_toolkits.axes_grid1 import make_axes_locatable


class CamRunner:
    def __init__(self, network, model_path, save_path, num_transformer_bins, transform=None, normalization=None,
                 regression_tasks=None, binary_cls_tasks=None, cls_tasks=None):
        """
        Creates a new instance of the CamRunner class.
        This class is designed to create and save CAM images for VIPA models.
        network - a string with the name of the network
        model_path - a string containing an absolute path to the model checkpoint
        save_path - a directory where generated CAM images should be saved
        num_transformer_bins - the number of transformer bins used by the model
        transform (optional) - transforms of the type torchvision.transforms to be applied to the input data
        normalization (optional) - normalization of the type torchvision.transforms.Normalize to be applied to the input data
        regression_tasks - a list of string identifiers for each regression task (or None if no such tasks exist)
        binary_cls_tasks - a list of string identifiers for each binary classification task (or None if no such tasks exist)
        cls_tasks - a list of string identifiers for each classification task (or None if no such tasks exist)
        """
        if regression_tasks is None:
            regression_tasks = dict()
        if binary_cls_tasks is None:
            binary_cls_tasks = dict()
        if cls_tasks is None:
            cls_tasks = dict()
        self.config = {'num_transformer_bins': num_transformer_bins,
                       'regression_tasks': regression_tasks,
                       'binary_cls_tasks': binary_cls_tasks,
                       'cls_tasks': cls_tasks,
                       'network': network,
                       'pretrained': True,
                       'gen_cam_map': True,
                       'attn_tasks': dict(),
                       'fine_tune_outputs': False,
                       'spectrogram': False,
                       'wavelet': False
                       }
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.save_path = save_path

        self.transform = transform
        self.normalization = normalization

        self.model = self.load_model(model_path)

    def get_average_cam_and_save(self, data, data_description, regression_task_index):
        """
        Generates and saves the average cam for a specified set of data and regression task.
        data - input data
        data_description - a string used to describe the data (to be used in the cam file name)
        regression_task_index - the index of the desired regression_task within the regression tasks list
        """
        maps = self.run_model(data)
        if data_description == "overall":
            norm_maps = []
            for map in maps:
                norm_map = (map-np.min(map))/(np.max(map)-np.min(map))
                norm_maps.append(norm_map)
            maps = norm_maps
        mean_map = np.mean(maps, axis=0)
        combined_mean_map = np.mean(mean_map, axis=1)
        dir_name = "mean_cams_" + self.config["regression_tasks"][regression_task_index]
        cam_filename = data_description + "-cam.jpg"
        augmented_image = self.plot_cam(data[0], mean_map, regression_task_index)
        image_filename = data_description + "-image-only.jpg"
        self.save_cam(augmented_image, dir_name, cam_filename)
        self.save_original_image(data[0], dir_name, image_filename)

    def load_model(self, model_path):
        """
        Loads the specified model checkpoint.
        model_path - string with the absolute path to the model checkpoint file
        @returns the loaded model
        """
        model = get_network(self.config)
        model.to(self.device)

        checkpoint = torch.load(model_path, map_location=torch.device(self.device))

        for key in list(checkpoint.keys()):
            if 'module.' in key:
                checkpoint[key.replace('module.', '')] = checkpoint[key]
                del checkpoint[key]
        model.load_state_dict(checkpoint)
        model.eval()

        return model

    def run_model(self, test_data):
        """
        Completes a forward pass on the model
        test_data - the data to be tested
        @returns class activation maps from the model
        """
        maps = []
        transform = transforms.ToTensor()
        self.model.eval()
        for data in test_data:
            if self.transform is not None:
                data = transform(data)
            else:
                data = torch.from_numpy(data).float()
            if self.normalization is not None:
                data = self.normalization(data)
            if self.config["network"] == "transformer":
                data = np.reshape(data, (len(data) // self.config['num_transformer_bins'],
                                         self.config['num_transformer_bins']))
            data = data.float()
            data = data.to(self.device)
            output, regression_maps = self.model.predict(data[np.newaxis, :])
            maps.append(regression_maps)
        return maps

    @staticmethod
    def plot_cam(image, cam, regression_task_index, force_resize=False, resize_size=224):
        """
        Creates an augmented image with the CAM overlaid on the original data
        image - the original image
        cam - the cam from the model's forward pass
        regression_task_index - the index of the desired regression task in the list of regression tasks
        force_resize (optional) - boolean indicating whether the image should be resized
        resize_size (optional) - the dimensions of the new image after resizing
        @returns the augmented image
        """

        if force_resize:
            source_images = F.interpolate(image, [resize_size, resize_size], mode='bicubic', align_corners=True).clamp(
                0., 1.)

        augmented_images = list()
        sim = image
        if cam.ndim > 3:
            att_map = np.squeeze(cam, axis=0)[regression_task_index] 
        else:
            att_map = np.squeeze(cam, axis=0)
        h, w = att_map.shape
        flipped_axes = (image.shape[1], image.shape[0])
        att_map = cv2.resize(att_map, flipped_axes, interpolation=cv2.INTER_CUBIC)
        min_value = np.min(np.min(att_map, axis=0, keepdims=True), axis=1, keepdims=True)
        max_value = np.max(np.max(att_map, axis=0, keepdims=True), axis=1, keepdims=True)
        print(min_value, max_value)
        att_map = (att_map - min_value) / (max_value - min_value)

        # to uint8
        sim_normalised = sim/sim.max()
        sim_uint8 = (sim_normalised * 255).astype(np.uint8)
        att_map_uint8 = (att_map * 255).astype(np.uint8)
        att_map_uint8 = cv2.applyColorMap(att_map_uint8, cv2.COLORMAP_JET)
        augmented = cv2.addWeighted(sim_uint8, 0.5, att_map_uint8, 0.5, 0.0)

        augmented = cv2.cvtColor(augmented, cv2.COLOR_BGR2RGB)
        augmented = np.moveaxis(augmented, (0, 1, 2), (1, 2, 0)) / 255.

        return augmented

    def save_original_image(self, image, dir_name, file_name):
        """
        Saves the original image to file.
        image - the image to be saved
        dir_name - a string containing the name of the directory where the image should be saved
        file_name - a string containing the name of the file
        """
        fig, ax = plt.subplots()
        img = plt.imshow(image, cmap="Greys")
        
        plt.axis('off')
        filepath = os.path.join(self.save_path, dir_name, file_name)
        plt.tight_layout(pad=0)
        plt.savefig(filepath, bbox_inches='tight', pad_inches=0)

    def save_cam(self, augmented_image, dir_name, file_name):
        """
        Saves the augmented image of CAM overlaid on the original data to file.
        augmented_image - the augmented image
        dir_name - a string containing the name of the directory where the image should be saved
        file_name - a string containing the name of the file
        """
        dname = '{}'.format(dir_name)
        dname = os.path.join(self.save_path, dname)
        if not os.path.exists(dname):
            os.makedirs(dname)

        fpath = os.path.join(dname, file_name)
        torchvision.utils.save_image(torch.tensor(augmented_image), fp=fpath,
                                     nrow=int(round(np.sqrt(len(augmented_image)))))

    @staticmethod
    def get_image_from_file(image_type, image_num, dir_name=''):
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
        filepath = os.path.join(dir_name, filepath)
        with open(filepath, 'rb') as file:
            return np.fromfile(file, np.int16).reshape((320, 256)).T

    def construct_image(self, image_num, dir):
        """
        Constructs a single VIPA image, ready for processing, from 4 VIPA images.
        The assumed 4 images are Sig/DarkSig/Ref/DarkRef
        image_num: image number of the desired image
        dir: path from self.root to the directory where the desired image is stored.
        @returns the image as a 3 x 320 x 256 numpy array
        """
        dark_ref = self.get_image_from_file("DarkRef", image_num + 1, dir)
        ref = self.get_image_from_file("Ref", image_num + 1, dir)
        dark_sig = self.get_image_from_file("DarkSig", image_num + 1, dir)
        sig = self.get_image_from_file("Sig", image_num + 1, dir)

        ref_corrected_image = ref - dark_ref
        sig_corrected_image = sig - dark_sig

        image = np.divide(sig_corrected_image, ref_corrected_image, out=np.zeros_like(sig_corrected_image, float),
                          where=ref_corrected_image != 0)

        # two of the images have outlier pixels, this clips them in line with other values
        image = np.clip(image, 0., 2.)

        # resnet operates on rgb images so add two additional channels that repeat the first channel
        image = np.repeat(image[:, :, np.newaxis], 3, axis=2)

        return image
