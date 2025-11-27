import torch
import torch.nn as nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class MLP(nn.Module):
    def __init__(self, config):
        """
        Creates a new instance of an MLP model. Hidden layer sizes are set by the h1, h2, and h3 params in config.
        config - a dictionary with all project config settings
        """
        super(MLP, self).__init__()
        self.config = config
        input_length = 12000
        #best results from bayesian optimistation
        h1_nodes = 9780
        h2_nodes = 939
        h3_nodes = 3976
        #h1_nodes = int(self.config["h1"])
        #h2_nodes = int(self.config["h2"])
        #h3_nodes = int(self.config["h3"])
        self.mlp = nn.Sequential(
            nn.Linear(input_length, h1_nodes),
            nn.ReLU(),
            nn.Linear(h1_nodes, h2_nodes),
            nn.ReLU(),
            nn.Linear(h2_nodes, h3_nodes),
            nn.ReLU()
        )
        modules = list()
        modules.append(nn.Flatten())
        modules.append(nn.Linear(h3_nodes, len(self.config['regression_tasks'])))
        self.regressor = nn.Sequential(*modules)

        modules = list()
        modules.append(nn.Flatten())
        modules.append(nn.Linear(h3_nodes, len(self.config['binary_cls_tasks'])))
        self.binary_classifier = nn.Sequential(*modules)

    def forward(self, input):
        """
        Completes a forward pass of the model.
        input - the input to the model. This should be in the form of a tensor with length config["sample_length"].
        @returns outputs of the model. Specifically, this includes logits for regression, binary classification,
        classification, and attention tasks
        """
        features = self.mlp(input)
        cls_logits = list()
        cls_maps = list()
        attn_logits = list()  # []
        regression_maps = None
        binary_cls_maps = None
        cls_maps = None

        regression_logits = self.regressor(features)
        binary_cls_logits = self.binary_classifier(features)

        return regression_logits, binary_cls_logits, cls_logits, attn_logits, \
            regression_maps, binary_cls_maps, cls_maps

    def predict(self, x):
        """
        Completes a forward pass of the model and returns a numpy array containing output logits
        x - the input to the model. This should be in the form of a numpy array with length config["sample_length"]
        @returns outputs of the model in a numpy array. This numpy array contains the logits for regression, binary
        classification, and classification tasks.
        """
        self.eval()
        xx = torch.tensor(x, dtype=torch.float32).to(device)
        with torch.no_grad():
            regression_logits, binary_cls_logits, cls_logits, _, _, _, _ = self.forward(xx)
        output = torch.cat((regression_logits, binary_cls_logits), 1)
        output = output.cpu()
        return output.numpy(), None