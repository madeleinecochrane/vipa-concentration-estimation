import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn import functional as F
from modelling.models.MLP import MLP
from modelling.models.CNN import CNN
from modelling.models.TransformerNet import TransformerNet

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Net(nn.Module):

    def __init__(self, config):
        """
         Creates a new instance of a pre-trained model for VIPA image analysis. Model backbone is as specified in the
         network setting in project config. ResNet, VGG and Swin based backbones are supported.
         config - a dictionary with all project config settings
         """
        super(Net, self).__init__()

        self.config = config

        if self.config['pretrained']:
            net = getattr(models, self.config['network'])(weights='IMAGENET1K_V1')
        else:
            net = getattr(models, self.config['network'])()

        if 'resnext' in self.config['network'] or 'resnet' in self.config['network']:
            in_features = list(net.fc.modules())[-1].in_features
        elif 'vgg' in self.config['network']:
            in_features = 512
        elif 'swin' in self.config['network']:
            in_features = net.head.in_features
        else:
            in_features = list(net.classifier.modules())[-1].in_features

        modules = list(net.children())
        if 'resnet' in self.config['network'] or 'resnext' in self.config['network']:
            modules = modules[:-2]
        elif 'densenet' in self.config['network'] or 'vgg' in self.config['network'] or \
                'mobilenet' in self.config['network']:
            modules = modules[:-1]
        elif 'swin' in self.config['network']:
            modules = modules[:-3]
        else:
            raise ValueError('Unknown backbone')

        self.net = nn.Sequential(*modules)

        # regression module
        modules = list()
        modules.append(nn.AdaptiveAvgPool2d((1, 1)))
        modules.append(nn.Flatten())
        modules.append(nn.Linear(in_features, len(self.config['regression_tasks'])))
        self.regressor = nn.Sequential(*modules)

        # binary classification module
        modules = list()
        modules.append(nn.AdaptiveAvgPool2d((1, 1)))
        modules.append(nn.Flatten())
        modules.append(nn.Linear(in_features, len(self.config['binary_cls_tasks'])))
        self.binary_classifier = nn.Sequential(*modules)

        # classification module
        self.classifiers = nn.ModuleDict()
        for t_name in self.config['cls_tasks']:
            modules = list()
            modules.append(nn.AdaptiveAvgPool2d((1, 1)))
            modules.append(nn.Flatten())
            modules.append(nn.Linear(in_features, self.config['cls_tasks'][t_name]))
            classifier = nn.Sequential(*modules)
            self.classifiers[t_name] = classifier

        self.gen_cam_map = self.config['gen_cam_map']

    def forward(self, input):
        """
        Completes a forward pass of the model.
        input - the input to the model. This should be in the form of a tensor with length config["sample_length"].
        @returns outputs of the model. Specifically, this includes logits for regression, binary classification,
        classification, and attention tasks
        """
        features = self.net(input)
        cls_logits = list()
        cls_maps = list()
        for c_name in self.classifiers:
            classifier = self.classifiers[c_name]

            # get CAM maps
            c = self.get_cam_faster(features, classifier)
            cls_maps.append(c)

            if len(self.config['attn_tasks']) != 0:
                # attention guidance computes soft weights rather than AVG
                a = torch.softmax(c.reshape(c.shape[0], c.shape[1], -1), dim=2).reshape(c.shape)
                cls_logits.append((c.contiguous() * a).sum(dim=(2, 3)))
            else:
                # AVG
                cls_logits.append(c.mean(dim=(2, 3)))

        regression_logits = self.regressor(features)
        regression_maps = self.get_cam_faster(features, self.regressor)

        binary_cls_logits = self.binary_classifier(features)
        binary_cls_maps = self.get_cam_faster(features, self.binary_classifier)

        cls_maps = list()
        for c_name in self.classifiers:
            if c_name in self.config['cls_tasks']:
                cls_maps.append(self.get_cam_faster(features, self.classifiers[c_name]))

        attn_logits = list()  # []
        if len(self.config['attn_tasks']) != 0:
            for a_name in self.config['attn_tasks']:
                c_name = self.config['attn_tasks'][a_name]['attn_of']
                cls_maps_idx = list(self.config['cls_tasks'].keys()).index(c_name)
                attn_logits.append(cls_maps[cls_maps_idx])

        return regression_logits, binary_cls_logits, cls_logits, attn_logits, \
            regression_maps, binary_cls_maps, cls_maps

    def predict(self, x):
        """
        Completes a forward pass of the model and returns a numpy array containing output logits
        x - the input to the model. This should be in the form of a 3-dimensional numpy array containing the VIPA image
        @returns outputs of the model in a numpy array. This numpy array contains the logits for regression and binary
        classification tasks as well as CAM maps for regression tasks.
        """
        self.eval()
        # x is numpy not tensor, return is numpy
        xx = torch.tensor(x, dtype=torch.float32).to(device)
        with torch.no_grad():
            regression_logits, binary_cls_logits, cls_logits, _, regression_maps, _, _ = self.forward(xx)
        output = torch.cat((regression_logits, binary_cls_logits), 1)
        output = output.cpu()
        if regression_maps is not None:
            regression_maps = regression_maps.cpu()
            return output.numpy(), regression_maps.numpy()
        return output.numpy()

    def get_cam_fast(self, features, classifier):
        """
        Creates a CAM map showing the regions of highest importance to the model
        features - the features after being passed through the model backbone for feature extraction
        classifier - the classifier layer corresponding to the desired task (i.e. regressor, binary classifier etc.)
        @returns activation maps - containing the CAM maps
        """
        if not self.gen_cam_map:
            return None

        cls_weights = classifier[-1].weight
        cls_bias = classifier[-1].bias

        cls_weights = cls_weights.permute(1, 0)
        cls_weights = cls_weights.view(1, cls_weights.shape[0], 1, 1, cls_weights.shape[1])
        act_maps = (features.view(list(features.shape) + [1]) * cls_weights).sum(dim=1)
        act_maps = act_maps.permute(0, 3, 1, 2) + cls_bias.view(1, -1, 1, 1)

        return act_maps

    def get_cam_faster(self, features, classifier):
        """
        Creates a CAM map showing the regions of highest importance to the model
        features - the features after being passed through the model backbone for feature extraction
        classifier - the classifier layer corresponding to the desired task (i.e. regressor, binary classifier etc.)
        @returns activation maps - containing the CAM maps
        """
        if not self.gen_cam_map:
            return None
        cls_weights = classifier[-1].weight
        cls_bias = classifier[-1].bias

        if cls_weights.shape[0] == 0:
            return self.get_cam_fast(features, classifier)
        else:
            act_maps = F.conv2d(features, cls_weights.view(cls_weights.shape[0], cls_weights.shape[1], 1, 1),
                                cls_bias, stride=1, padding=0, dilation=1, groups=1)

            return act_maps


def get_network(config):
    if config['network'] == "mlp":
        return MLP(config)
    elif config['network'] == "transformer":
        return TransformerNet(config)
    elif config['network'] == "cnn":
        return CNN(config)
    else:
        return Net(config)
