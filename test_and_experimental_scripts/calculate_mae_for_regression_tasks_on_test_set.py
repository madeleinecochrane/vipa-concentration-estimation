import os
import pickle
import numpy as np
import torch
from modelling.model import get_network
from torchvision import transforms

device = 'cuda' if torch.cuda.is_available() else 'cpu'

config = {'num_transformer_bins': 1000,
          'regression_tasks': ['reg1', 'reg2'],
          'binary_cls_tasks': ['bc1', 'bc2'],
          'cls_tasks': dict(),
          'network': 'transformer',
          'pretrained': True,
          'gen_cam_map': False,
          'attn_tasks': dict()
          }

network = "<REPLACE THIS WITH YOUR DESIRED NETWORK e.g. cnn>"
model_path = "<REPLACE THIS WITH THE PATH TO YOUR MODEL CHECKPOINT>"
data_path = "<REPLACE THIS WITH THE PATH TO YOUR TEST PKL FILE>" 

def load_model(network, model_path):
    if not os.path.exists(model_path):
        print("Provided model path does not exist")
        return None
    config.update({'network': network})
    model = get_network(config)
    model.to(device)
    model.eval()
    checkpoint = torch.load(model_path)
    for key in list(checkpoint.keys()):
        if 'module.' in key:
            checkpoint[key.replace('module.', '')] = checkpoint[key]
            del checkpoint[key]
    model.load_state_dict(checkpoint)
    return model

def load_data_and_labels(network, data_path):
    if not os.path.exists(data_path):
        print("Provided data path does not exist")
        return None, None
    test_data, test_labels = pickle.load(open(data_path, 'rb'))
    return test_data, test_labels

def get_task_label_by_index(index):
    if index < len(config["regression_tasks"]):
        return config["regression_tasks"][index]
    elif index < len(config["regression_tasks"] + config["binary_cls_tasks"]):
        return config["binary_cls_tasks"][index - len(config["regression_tasks"])]
    else:
        return config["cls_tasks"][index - len(config["regression_tasks"]) - len(config["binary_cls_tasks"])]


model = load_model(network, model_path)
test_data, test_labels = load_data_and_labels(network, data_path)

num_regression_tasks = len(config["regression_tasks"])
is_spectral_analysis = network == "cnn" or network == "transformer" or network == "mlp"

all_absolute_differences = []

if model is not None and test_data is not None:
    if is_spectral_analysis:
        transform = None
        normalization = None
    else:
        transform = transforms.ToTensor()
        normalization = transforms.Normalize(mean=(0.98, 0.98, 0.98), std=(0.0761, 0.0761, 0.0761)) # change these normalization values to match your own dataset

    for i in range(len(test_data)):
        data = test_data[i]
        if transform is not None:
            data = transform(data)
        if normalization is not None:
            data = normalization(data)
        if not is_spectral_analysis:
            data = data.float()
        labels = test_labels[i][:num_regression_tasks]

        output = model.predict(data[np.newaxis,:])
        preds = []
        for j in range(num_regression_tasks):
            if is_spectral_analysis:
                preds.append(output[0][0][j])
            else:
                preds.append(output[0][j])
        preds = np.array(preds)
        
        absolute_differences = np.abs(preds - labels)
        all_absolute_differences.append(absolute_differences)

    all_absolute_differences = np.array(all_absolute_differences)
    mean_absolute_differences = np.mean(all_absolute_differences, axis=0)
    for j in range(num_regression_tasks):
        task_label = get_task_label_by_index(j)
        print("MAE for", task_label, "is", str(mean_absolute_differences[j]))

