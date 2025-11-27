import argparse
import numpy as np
import os
import torch
import torchvision.transforms.v2 as transforms
from torch.utils.data import DataLoader
from modelling.trainer import Trainer
from modelling.model import get_network
from modelling.datasets.IsotopicCo2Spectra import IsotopicCo2Spectra
from modelling.datasets.IsotopicCo2Images import IsotopicCo2Images
from modelling.recorder import Recorder
from toolbox.general_utils import str2bool
from toolbox.save_utils import save_mat
from toolbox.json_utils import save_json
from path_utils import dataset_root, data_root

device = "cuda" if torch.cuda.is_available() else 'cpu'


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.save_path, exist_ok=True)

    if 'VIPA' in args.data_path:
        print('Experiments for VIPA images')
        args.input_size = 3000
        train_transform = transforms.ToTensor()
        val_transform = transforms.ToTensor()
    elif 'IsotopicCo2Spectra' in args.data_path:
        print('Experiments for Isotopic CO2 Spectra')
        args.input_size = 3000
        train_transform = None
        val_transform = None
    else:
        raise ValueError('Unknown dataset. data path is ' + args.data_path)

    #print(args)

    config = {'gen_cam_map': args.cam_interval != -1,
              'network': args.network,
              'pretrained': args.pretrained,
              'num_training_images_per_class': args.num_training_images_per_class,
              'cls_tasks': {},
              'ignore_value': -1e10,
              'args': vars(args),
              'num_transformer_bins': args.num_transformer_bins,
              'num_training_images': args.num_training_images,
              'num_translations': args.num_translations,
              'noise_factor': args.noise_factor,
              'h1': args.h1,
              'h2': args.h2,
              'h3': args.h3,
              'd1': args.d1,
              'd2': args.d2,
              'd3': args.d3,
              'verbose': args.verbose,
              }

    if 'VIPA' in args.data_path:
        config['regression_tasks'] = ['reg1', 'reg2']
        config['binary_cls_tasks'] = ['12CO2', '13CO2']
        config['cls_tasks'] = dict()
        train_dataset = IsotopicCo2Images(train=True, transform=train_transform, root=args.data_path, config=config)
        train_data_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True,
                                       num_workers=args.num_workers, pin_memory=True)
        val_dataset = IsotopicCo2Images(train=False, transform=val_transform, root=args.data_path, config=config)
        val_data_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=False,
                                     num_workers=args.num_workers, pin_memory=True)

    if 'IsotopicCo2Spectra' in args.data_path:
        config['regression_tasks'] = ['reg1', 'reg2']
        config['binary_cls_tasks'] = ['12CO2', '13CO2']
        config['cls_tasks'] = dict()
        train_dataset = IsotopicCo2Spectra(train=True, transform=train_transform, root=args.data_path, config=config)
        train_data_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True,
                                       num_workers=args.num_workers, pin_memory=True)
        val_dataset = IsotopicCo2Spectra(train=False, transform=val_transform, root=args.data_path, config=config)
        val_data_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=False,
                                     num_workers=args.num_workers, pin_memory=True)


    config['attn_tasks'] = dict()
    save_json(os.path.join(args.save_path, 'config.json'), config)

    net = get_network(config)

    # reload trained model weights from a checkpoint
    if args.reload_from_checkpoint:
        if config["verbose"]:
            print('Loading from checkpoint: {}'.format(args.reload_path))
        if os.path.exists(args.reload_path):
            net.load_state_dict(torch.load(args.reload_path))
        else:
            print('File not exists in the reload path: {}'.format(args.reload_path))

    if 'cuda' in device:
        if config["verbose"]:
            print('Multiple GPUS.......')
        net = torch.nn.DataParallel(net).to(device)

    params = list(net.parameters())
    if config["verbose"]:
        print('Trainable parameters:', sum(p.numel() for p in net.parameters() if p.requires_grad))
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(net.parameters(), lr=args.learning_rate, weight_decay=1e-4, momentum=0.9)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(params, lr=args.learning_rate, weight_decay=5e-4)
    else:
        raise ValueError('Unexpected optimizer: {}'.format(args.optimizer))

    schedule = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.learning_rate_step_size,
                                               gamma=args.learning_rate_gamma)
    recorder = Recorder()


    trainer = Trainer(
        net, optimizer, schedule, recorder, train_data_loader, val_data_loader, config, args, device
    )
    recorder = trainer.train()
    save_mat(args.save_path, recorder.master_dict)
    return recorder


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    torch.autograd.set_detect_anomaly(True)
    parser.add_argument('--data_path', type=str, default=os.path.join(dataset_root, 'IsotopicCo2Spectra'),
                        help='the path to data')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--num_epochs', type=int, default=95)
    parser.add_argument('--save_path', type=str, default=os.path.join(data_root, 'results'))
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
    parser.add_argument('--num_training_images_per_class', type=int, default=5)
    parser.add_argument('--num_transformer_bins', type=int, default=100)
    parser.add_argument('--limited_training_data', type=str2bool, default='False')
    parser.add_argument('--num_training_images', type=int, default=2400)
    parser.add_argument('--num_translations', type=int, default=0)
    parser.add_argument('--noise_factor', type=float, default=0.0)
    parser.add_argument('--h1', type=int, default=5000)
    parser.add_argument('--h2', type=int, default=2000)
    parser.add_argument('--h3', type=int, default=1000)
    parser.add_argument('--d1', type=float, default=0)
    parser.add_argument('--d2', type=float, default=0)
    parser.add_argument('--d3', type=float, default=0)
    parser.add_argument('--verbose', type=str2bool, default='True')

    args = parser.parse_args()
    main(args)
