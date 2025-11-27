import torch
import torch.nn as nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class TransformerNet(nn.Module):
    def __init__(self, config):
        """
        Creates a new instance of a Transformer model. Model dimensions are set according to the number of transformer
        bins specified in the project config
        config - a dictionary with all project config settings
        """
        super(TransformerNet, self).__init__()
        self.config = config
        num_transformer_layers = 6
        dim_model = self.config['num_transformer_bins']
        num_heads = 4
        dropout = 0.0
        hidden_features = self.config['num_transformer_bins']*2
        transformer_layer = nn.TransformerEncoderLayer(d_model=dim_model,nhead=num_heads,
                                                        dim_feedforward=hidden_features, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(transformer_layer, num_layers=num_transformer_layers)

        if len(self.config['regression_tasks']):
            modules = list()
            modules.append(nn.Linear(dim_model, len(self.config['regression_tasks'])))
            self.regressor = nn.Sequential(*modules)

        modules = list()
        modules.append(nn.Linear(dim_model, len(self.config['binary_cls_tasks'])))
        self.binary_classifier = nn.Sequential(*modules)

        self.classifiers = nn.ModuleDict()
        for t_name in self.config['cls_tasks']:
            modules = list()
            modules.append(nn.Linear(dim_model, self.config['cls_tasks'][t_name]))
            classifier = nn.Sequential(*modules)
            self.classifiers[t_name] = classifier

    def forward(self, input):
        """
        Completes a forward pass of the model.
        input - the input to the model. This should be in the form of a tensor with length config["sample_length"].
        @returns outputs of the model. Specifically, this includes logits for regression, binary classification,
        classification, and attention tasks
        """
        features = self.transformer_encoder(input)
        regression_logits = list()
        binary_cls_logits = list()
        cls_logits = list()
        cls_maps = list()
        attn_logits = list()
        regression_maps = None
        binary_cls_maps = None

        features = features.mean(dim=1)

        if len(self.config['regression_tasks']):
            regression_logits = self.regressor(features)
        binary_cls_logits = self.binary_classifier(features)
        for c_name in self.classifiers:
            classifier = self.classifiers[c_name]
            cls_logits.append(classifier(features))

        return regression_logits, binary_cls_logits, cls_logits, attn_logits, \
            regression_maps, binary_cls_maps, cls_maps

    def predict(self, x):
        """
        Completes a forward pass of the model and returns a numpy array containing output logits
        x - the input to the model. This should be in the form of a numpy array with three dimensions - the first equal
        to the batch size
        @returns outputs of the model in a numpy array. This numpy array contains the logits for regression and binary
        classification tasks.
        """
        self.eval()
        first_dim = x.shape[0]
        xx = torch.tensor(x, dtype=torch.float32).to(device)
        if x.ndim == 3:
            all_points = x.shape[1] * x.shape[2]
        else:
            all_points = x.shape[1]
        xx = torch.reshape(xx, (first_dim, all_points // self.config['num_transformer_bins'],
                                self.config['num_transformer_bins']))  # reshape to 100 x 120

        with torch.no_grad():
            regression_logits, binary_cls_logits, cls_logits, _, _, _, _ = self.forward(xx)
        output = torch.cat((regression_logits, binary_cls_logits), 1)
        output = output.cpu()
        return output.numpy(), None
