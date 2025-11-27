import os
import torch
import argparse
import numpy as np
from modelling.train import main
from toolbox.general_utils import str2bool
from bayes_opt import BayesianOptimization
from path_utils import dataset_root, data_root


torch.autograd.set_detect_anomaly(True)
test_summary = []


def evaluate_network(layer_1, layer_2, layer_3):
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', type=str, default=os.path.join(dataset_root, 'IsotopicCo2Spectra'),
                        help='the path to data')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--num_epochs', type=int, default=95)
    parser.add_argument('--save_path', type=str, default=os.path.join(data_root, 'results_2'))
    parser.add_argument('--save_interval', type=int, default=10, help='#epochs')
    parser.add_argument('--cam_interval', type=int, default=-1, help='#batches')
    parser.add_argument('--display_interval', type=int, default=100, help='#batches')

    parser.add_argument('--reload_path', type=str, default='NA', help='path for trained network')
    parser.add_argument('--reload_from_checkpoint', type=str2bool, default='False')

    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--optimizer', type=str, default='sgd')
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--learning_rate_step_size', type=int, default=30)
    parser.add_argument('--learning_rate_gamma', type=str, default=0.1)

    parser.add_argument('--network', type=str, default='mlp')
    parser.add_argument('--pretrained', type=str2bool, default='True')
    parser.add_argument('--enable_attn_loss', type=str2bool, default='False')
    parser.add_argument('--attn_loss_ver', type=int, default=8)
    parser.add_argument('--attn_loss_scalar', type=float, default=0.1)
    parser.add_argument('--num_training_images_per_class', type=int, default=5)
    parser.add_argument('--num_transformer_bins', type=int, default=100)
    parser.add_argument('--num_training_images', type=int, default=2400)
    parser.add_argument('--limited_training_data', type=str2bool, default=False)
    parser.add_argument('--corrected_image', type=str2bool, default='False')
    parser.add_argument('--embedding_strat', type=str, default='none')
    parser.add_argument('--num_translations', type=int, default=0)
    parser.add_argument('--noise_factor', type=float, default=0.0)
    parser.add_argument('--d1', type=float, default=0)
    parser.add_argument('--d2', type=float, default=0)
    parser.add_argument('--d3', type=float, default=0)
    parser.add_argument('--h1', type=int, default=layer_1)
    parser.add_argument('--h2', type=int, default=layer_2)
    parser.add_argument('--h3', type=int, default=layer_3)
    parser.add_argument('--verbose', type=str2bool, default='False')
    args = parser.parse_args()
    recorder = main(args)
    master_dict = recorder.master_dict
    final_epoch = master_dict["e95"]
    val = final_epoch["val"]
    loss_reg1 = val["loss_reg1"]
    mean_loss = np.mean(loss_reg1)
    summary_string = "Score: " + str(mean_loss) + " H1: " + str(layer_1) + " H2: " + str(layer_2) + " H3: " + str(layer_3)
    test_summary.append(summary_string)
    return mean_loss*-1


pbounds = {'layer_1': (1, 10000),
           'layer_2': (1, 10000),
           'layer_3': (1, 10000)}

optimiser = BayesianOptimization(
    f=evaluate_network,
    pbounds=pbounds,
    verbose=2,
    random_state=1
)

optimiser.maximize(init_points=200, n_iter=400)
for summary_string in test_summary:
    print(summary_string)
print(optimiser.max)
