import torch
import torch.nn as nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class CNN(nn.Module):
    def __init__(self, config):
        """
        Creates a new instance of a CNN model. Change channel sizes here if you want to customise for your own data.
        config - a dictionary with all project config settings
        """
        super(CNN, self).__init__()
        self.config = config
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=120, out_channels=100, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=100, out_channels=50, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        modules = list()
        modules.append(nn.Flatten())
        modules.append(nn.Linear(1100, len(self.config['regression_tasks'])))
        self.regressor = nn.Sequential(*modules)

        modules = list()
        modules.append(nn.Flatten())
        modules.append(nn.Linear(1100, len(self.config['binary_cls_tasks'])))
        self.binary_classifier = nn.Sequential(*modules)

    def forward(self, input):
        """
        Completes a forward pass of the model.
        input - the input to the model. This should be in the form of a tensor with length config["sample_length"].
        @returns outputs of the model. Specifically, this includes logits for regression, binary classification,
        classification, and attention tasks
        """
        # TODO: change line below if necessary to tailor to your data
        input = torch.reshape(input, (int((input.shape[0]*input.shape[1])/12000), 120, 100))
        features = self.cnn(input)
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
        self.eval()
        xx = torch.tensor(x, dtype=torch.float32).to(device)
        with torch.no_grad():
            regression_logits, binary_cls_logits, cls_logits, _, regression_maps, _, _ = self.forward(xx)
        output = torch.cat((regression_logits, binary_cls_logits), 1)
        output = output.cpu()
        return output.numpy(), None