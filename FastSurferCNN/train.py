# Copyright 2019 Image Analysis Lab, German Center for Neurodegenerative Diseases (DZNE), Bonn
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# IMPORTS
import argparse
import os
import shutil
import logging

import torch
import torch.nn as nn

from torch.utils.data.dataloader import DataLoader
from torchvision import transforms, utils

from data_loader.load_neuroimaging_data import AsegDatasetWithAugmentation
from data_loader.augmentation import AugmentationPadImage, AugmentationRandomCrop, ToTensor

from models.networks import FastSurferCNN
from models.solver import Solver

##
# GLOBAL VARS
##
CLASS_NAMES_SAG = ["3rd-Ventricle", "4th-Ventricle", "Brain-Stem", "CSF", "Cerebral-White-Matter",
                   "Lateral-Ventricle", "Inf-Lat-Vent", "Cerebellum-White-Matter",
                   "Cerebellum-Cortex", "Thalamus-Proper", "Caudate", "Putamen",
                   "Pallidum", "Hippocampus", "Amygdala", "Accumbens-area",
                   "VentralDC", "choroid-plexus", "WM-hypointensities",
                   "ctx-both-caudalanteriorcingulate", "ctx-both-caudalmiddlefrontal",
                   "ctx-both-cuneus", "ctx-both-entorhinal", "ctx-both-fusiform",
                   "ctx-both-inferiorparietal", "ctx-both-inferiortemporal", "ctx-both-isthmuscingulate",
                   "ctx-both-lateraloccipital", "ctx-both-lateralorbitofrontal", "ctx-both-lingual",
                   "ctx-both-medialorbitofrontal", "ctx-both-middletemporal", "ctx-both-parahippocampal",
                   "ctx-both-paracentral", "ctx-both-parsopercularis", "ctx-both-parsorbitalis",
                   "ctx-both-parstriangularis", "ctx-both-pericalcarine", "ctx-both-postcentral",
                   "ctx-both-posteriorcingulate", "ctx-both-precentral", "ctx-both-precuneus",
                   "ctx-both-rostralanteriorcingulate", "ctx-both-rostralmiddlefrontal", "ctx-both-superiorfrontal",
                   "ctx-both-superiorparietal", "ctx-both-superiortemporal", "ctx-both-supramarginal",
                   "ctx-both-transversetemporal", "ctx-both-insula"
                   ]

CLASS_NAMES = ["Left-Cerebral-White-Matter", "Left-Lateral-Ventricle", "Left-Inf-Lat-Vent",
               "Left-Cerebellum-White-Matter", "Left-Cerebellum-Cortex", "Left-Thalamus-Proper",
               "Left-Caudate", "Left-Putamen", "Left-Pallidum", "3rd-Ventricle", "4th-Ventricle",
               "Brain-Stem", "Left-Hippocampus", "Left-Amygdala", "CSF", "Left-Accumbens-area",
               "Left-VentralDC", "Left-choroid-plexus", "Right-Cerebral-White-Matter",
               "Right-Lateral-Ventricle", "Right-Inf-Lat-Vent", "Right-Cerebellum-White-Matter",
               "Right-Cerebellum-Cortex", "Right-Thalamus-Proper", "Right-Caudate", "Right-Putamen",
               "Right-Pallidum", "Right-Hippocampus", "Right-Amygdala", "Right-Accumbens-area",
               "Right-VentralDC", "Right-choroid-plexus", "WM-hypointensities", "ctx-lh-caudalanteriorcingulate",
               "ctx-both-caudalmiddlefrontal", "ctx-lh-cuneus", "ctx-both-entorhinal", "ctx-both-fusiform",
               "ctx-both-inferiorparietal", "ctx-both-inferiortemporal", "ctx-lh-isthmuscingulate",
               "ctx-both-lateraloccipital", "ctx-lh-lateralorbitofrontal", "ctx-lh-lingual",
               "ctx-lh-medialorbitofrontal", "ctx-both-middletemporal", "ctx-lh-parahippocampal",
               "ctx-lh-paracentral", "ctx-both-parsopercularis", "ctx-both-parsorbitalis",
               "ctx-both-parstriangularis", "ctx-lh-pericalcarine", "ctx-lh-postcentral",
               "ctx-lh-posteriorcingulate", "ctx-lh-precentral", "ctx-lh-precuneus",
               "ctx-both-rostralanteriorcingulate", "ctx-both-rostralmiddlefrontal", "ctx-lh-superiorfrontal",
               "ctx-both-superiorparietal", "ctx-both-superiortemporal", "ctx-both-supramarginal",
               "ctx-both-transversetemporal", "ctx-both-insula", "ctx-rh-caudalanteriorcingulate",
               "ctx-rh-cuneus", "ctx-rh-isthmuscingulate", "ctx-rh-lateralorbitofrontal", "ctx-rh-lingual",
               "ctx-rh-medialorbitofrontal", "ctx-rh-parahippocampal", "ctx-rh-paracentral",
               "ctx-rh-pericalcarine", "ctx-rh-postcentral", "ctx-rh-posteriorcingulate", "ctx-rh-precentral",
               "ctx-rh-precuneus", "ctx-rh-superiorfrontal"]


def setup_options():
    # Training settings
    parser = argparse.ArgumentParser(description='Segmentation')

    parser.add_argument('--batch_size', type=int, default=16, metavar='N',
                        help='input batch size for training (default: 16)')
    parser.add_argument('--validation_batch_size', type=int, default=16, metavar='N',
                        help='input batch size for validation (default: 16)')

    parser.add_argument('--hdf5_name_train', type=str,
                        help='path to training hdf5-dataset')
    parser.add_argument('--hdf5_name_val', type=str,
                        help='path to validation hdf5-dataset')

    parser.add_argument('--plane', type=str, default="axial", choices=["axial", "coronal", "sagittal"],
                        help="Which plane to train on (axial (default), coronal or sagittal)")
    parser.add_argument('--img_height', dest="height", type=int, default=256,
                        help='Height of image instances as returned by hdf5_loader (Default 256)')
    parser.add_argument('--img_width', dest="width", type=int, default=256,
                        help='Width of image instances as returned by hdf5_loader (Default 256)')

    parser.add_argument('--log-interval', type=int, default=2, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--log_dir', type=str,
                        help='log directory for run')

    parser.add_argument('--num_filters', type=int, default=64,
                        help='Filter dimensions for DenseNet (all layers same). Default=64')
    parser.add_argument('--num_classes', type=int, default=79,
                        help='Number of classes to predict, including background. Default=79')
    parser.add_argument('--num_channels', type=int, default=7,
                        help='Number of input channels. Default=7 (thick slices)')

    parser.add_argument('--kernel_height', type=int, default=5, help='Height of Kernel (Default 5)')
    parser.add_argument('--kernel_width', type=int, default=5, help='Width of Kernel (Default 5)')
    parser.add_argument('--stride', type=int, default=1, help="Stride during convolution (Default 1)")
    parser.add_argument('--stride_pool', type=int, default=2, help="Stride during pooling (Default 2)")
    parser.add_argument('--pool', type=int, default=2, help='Size of pooling filter (Default 2)')

    parser.add_argument('--epochs', type=int, default=30, metavar='N', help='number of epochs to train (default: 30)')
    parser.add_argument('--lr', type=float, default=1e-2, metavar='LR', help='learning rate (default: 0.01)')
    parser.add_argument('--optim', type=str, default="adam", choices=["adam", "sgd"])
    parser.add_argument('--resume', action="store_true", default=False, help="Flag if resume is needed")
    parser.add_argument('--torchv11', action="store_true", default=False,
                        help="Flag if torch version below 1.2 is used."
                             " Order of learning rate schedule update and optimizer step is changed between the versions.")
    parser.add_argument('--scheduler', type=str, default="StepLR", choices=["StepLR", "None"],
                        help="type of learning rate scheduler to use. Default: StepLR")
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for optimizer (SGD).')
    parser.add_argument('--nesterov', action='store_true', default=False, help='Enables Nesterov for optimizer (SGD)')

    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')

    args = parser.parse_args()
    return args


def train():
    """
    Function to train a network with the given input parameters
    :return: None
    """
    args = setup_options()

    # logger and snapshot current code
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    shutil.copy2(__file__, os.path.join(args.log_dir, "train.py"))
    shutil.copy2("./models/networks.py", os.path.join(args.log_dir, "networks.py"))
    shutil.copy2("./models/sub_module.py", os.path.join(args.log_dir, "sub_module.py"))

    logger = logging.getLogger("train")
    logger.setLevel(logging.DEBUG)
    logger.handlers = []
    ch = logging.StreamHandler()
    logger.addHandler(ch)
    fh = logging.FileHandler(os.path.join(args.log_dir, "log.txt"))
    logger.addHandler(fh)

    logger.info("%s", repr(args))

    # Define augmentations
    transform_train = transforms.Compose([AugmentationPadImage(pad_size=8),
                                          AugmentationRandomCrop(output_size=(args.height, args.width)),
                                          ToTensor()])
    transform_test = transforms.Compose([ToTensor()])

    # Prepare and load data
    params_dataset_train = {'dataset_name': args.hdf5_name_train, 'plane': args.plane}
    params_dataset_test = {'dataset_name': args.hdf5_name_val, 'plane': args.plane}

    dataset_train = AsegDatasetWithAugmentation(params_dataset_train, transforms=transform_train)
    dataset_validation = AsegDatasetWithAugmentation(params_dataset_test, transforms=transform_test)

    train_dataloader = DataLoader(dataset=dataset_train, batch_size=args.batch_size, shuffle=True)
    validation_dataloader = DataLoader(dataset=dataset_validation, batch_size=args.validation_batch_size, shuffle=True)

    # Set up network
    params_network = {'num_channels': args.num_channels, 'num_filters': args.num_filters,
                      'kernel_h': args.kernel_height, 'kernel_w': args.kernel_width, 'stride_conv': args.stride,
                      'pool': args.pool, 'stride_pool': args.stride_pool, 'num_classes': args.num_classes,
                      'kernel_c': 1, 'kernel_d': 1, 'batch_size': args.batch_size,
                      'height': args.height, 'width': args.width}

    # Select the model
    model = FastSurferCNN(params_network)

    # Put model on GPU
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model.cuda()

    # Define labels
    if args.plane == "sagittal":
        curr_labels = CLASS_NAMES_SAG

    else:
        curr_labels = CLASS_NAMES

    # optimizer selection
    if args.optim == "sgd":
        optim = torch.optim.SGD

        # Global optimization parameters for adam
        default_optim_args = {"lr": args.lr,
                              "momentum": args.momentum,
                              "dampening": 0,
                              "weight_decay": 0,
                              "nesterov": args.nesterov}
    else:
        optim = torch.optim.Adam

        # Global optimization parameters for adam
        default_optim_args = {"lr": args.lr,
                              "betas": (0.9, 0.999),
                              "eps": 1e-8,
                              "weight_decay": 0.0001}

    # Run network
    solver = Solver(num_classes=params_network["num_classes"], optimizer_args=default_optim_args, optimizer=optim)
    solver.train(model, train_dataloader, validation_dataloader, class_names=curr_labels, num_epochs=args.epochs,
                 log_params={'logdir': os.path.join(args.log_dir, "logs"), 'log_iter': args.log_interval, 'logger': logger},
                 expdir=os.path.join(args.log_dir, "ckpts"), scheduler_type=args.scheduler, torch_v11=args.torchv11,
                 resume=args.resume)


if __name__ == "__main__":
    train()
