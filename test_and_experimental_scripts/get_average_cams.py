import os
import pickle
from modelling.cam_runner import CamRunner
from torchvision import transforms


def separate_data_by_label(data, labels):
    c12o2_ratio = 0.9842
    concentration_levels = [0, 10, 25, 50, 75, 100]
    scaled_concentration_levels = [val * c12o2_ratio for val in concentration_levels]
    sep_data = {0: [], 10: [], 25: [], 50: [], 75: [], 100: []}
    for i in range(len(labels)):
        label = labels[i]
        c12_label = label[0]
        datapoint = data[i]
        concentration_index = scaled_concentration_levels.index(c12_label)
        sep_data[concentration_levels[concentration_index]].append(datapoint)
    return sep_data


network = 'resnet50'
save_path = '<REPLACE WITH PATH TO YOUR MODEL DIRECTORY>'
model_path = os.path.join(save_path, 'net_e95.ckpt')
num_transformer_bins = 500
regression_tasks = ['reg1', 'reg2']
binary_cls_tasks = ['12co2', '13co2']
transform = transforms.ToTensor()
normalization = transforms.Normalize(mean=(0.98, 0.98, 0.98), std=(0.0761, 0.0761, 0.0761))
cam_runner = CamRunner(network, model_path, save_path, num_transformer_bins, transform, normalization, regression_tasks, binary_cls_tasks)

root = "<REPLACE WITH PATH TO YOUR DATASET DIRECTORY>"
test_data, test_labels = pickle.load(open(os.path.join(root, 'processed/test.pkl'), 'rb'))

separated_data = separate_data_by_label(test_data, test_labels)

for task in len(regression_tasks):
    for key, value in separated_data.items():
        data_description = network + "-" + str(key) + "pc"
        cam_runner.get_average_cam_and_save(value, data_description, task)

    cam_runner.get_average_cam_and_save(test_data, data_description="overall", regression_task_index=task)


