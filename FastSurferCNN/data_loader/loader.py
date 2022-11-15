
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
from torchvision import transforms
from torch.utils.data import DataLoader
import torchio as tio

from FastSurferCNN.data_loader import dataset as dset
from FastSurferCNN.data_loader.augmentation import ToTensor, ZeroPad2D, AddGaussianNoise
from FastSurferCNN.utils import logging

logger = logging.getLogger(__name__)


def get_dataloader(cfg, mode):
    """
        Creating the dataset and pytorch data loader
    :param cfg:
    :param mode: loading data for train, val and test mode
    :return:
    """
    assert mode in ['train', 'val'], f"dataloader mode is incorrect {mode}"

    padding_size = cfg.DATA.PADDED_SIZE

    if mode == 'train':

        if "None" in cfg.DATA.AUG:
            tfs = [ZeroPad2D((padding_size, padding_size)), ToTensor()]
            # old transform
            if "Gaussian" in cfg.DATA.AUG:
                tfs.append(AddGaussianNoise(mean=0, std=0.1))

            data_path = cfg.DATA.PATH_HDF5_TRAIN
            shuffle = True

            logger.info(f"Loading {mode.capitalize()} data ... from {data_path}. Using standard Aug")

            dataset = dset.MultiScaleDatasetVal(data_path, cfg, transforms.Compose(tfs))
        else:

            # Elastic
            elastic = tio.RandomElasticDeformation(num_control_points=7,
                                                   max_displacement=(20, 20, 0),
                                                   locked_borders=2,
                                                   image_interpolation='linear',
                                                   include=['img', 'label', 'weight'])
            # Scales
            scaling = tio.RandomAffine(scales=(0.8, 1.15),
                                       degrees=0,
                                       translation=(0, 0, 0),
                                       isotropic=True,  # If True, scaling factor along all dimensions is the same
                                       center='image',
                                       default_pad_value='minimum',
                                       image_interpolation='linear',
                                       include=['img', 'label', 'weight'])

            # Rotation
            rot = tio.RandomAffine(scales=(1.0, 1.0),
                                   degrees=10,
                                   translation=(0, 0, 0),
                                   isotropic=True,  # If True, scaling factor along all dimensions is the same
                                   center='image',
                                   default_pad_value='minimum',
                                   image_interpolation='linear',
                                   include=['img', 'label', 'weight'])

            # Translation
            tl = tio.RandomAffine(scales=(1.0, 1.0),
                                  degrees=0,
                                  translation=(15.0, 15.0, 0),
                                  isotropic=True,  # If True, scaling factor along all dimensions is the same
                                  center='image',
                                  default_pad_value='minimum',
                                  image_interpolation='linear',
                                  include=['img', 'label', 'weight'])

            # Random Anisotropy (Downsample image along an axis, then upsample back to initial space
            ra = tio.transforms.RandomAnisotropy(axes=(0, 1),
                                                 downsampling=(1.1, 1.5),
                                                 image_interpolation="linear",
                                                 include=["img"])

            # Bias Field
            bias_field = tio.transforms.RandomBiasField(coefficients=0.5, order=3, include=['img'])

            # Gamma
            random_gamma = tio.transforms.RandomGamma(log_gamma=(-0.1, 0.1), include=['img'])

            #

            all_augs = {"Elastic": elastic, "Scaling": scaling, "Rotation": rot, "Translation": tl,
                        "RAnisotropy": ra, "BiasField": bias_field, "RGamma": random_gamma}

            all_tfs = {all_augs[aug]: 0.8 for aug in cfg.DATA.AUG if aug != "Gaussian"}
            gaussian_noise = True if "Gaussian" in cfg.DATA.AUG else False

            transform = tio.Compose([tio.Compose(all_tfs, p=0.8)], include=["img", "label", "weight"])

            data_path = cfg.DATA.PATH_HDF5_TRAIN
            shuffle = True

            logger.info(f"Loading {mode.capitalize()} data ... from {data_path}. Using torchio Aug")

            dataset = dset.MultiScaleDataset(data_path, cfg, gaussian_noise, transform)

    elif mode == 'val':
        data_path = cfg.DATA.PATH_HDF5_VAL
        shuffle = False
        transform = transforms.Compose([ZeroPad2D((padding_size, padding_size)),
                                        ToTensor(),
                                        ])

        logger.info(f"Loading {mode.capitalize()} data ... from {data_path}")

        dataset = dset.MultiScaleDatasetVal(data_path, cfg, transform)

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        num_workers=cfg.TRAIN.NUM_WORKERS,
        shuffle=shuffle,
        pin_memory=True,
    )
    return dataloader
