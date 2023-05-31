# Copyright 2022 Image Analysis Lab, German Center for Neurodegenerative Diseases (DZNE), Bonn
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

from FastSurferCNN.utils import logging

from CerebNet.data_loader import dataset as dset
from CerebNet.data_loader.augmentation import ToTensor, get_transform

logger = logging.get_logger(__name__)


def get_dataloader(cfg, mode):
    """
    Creating the dataset and pytorch data loader

    Args:
        cfg:
        mode: loading data for train, val and test mode

    Returns:
        the Dataloader
    """

    if mode == "train":
        data_path = cfg.DATA.PATH_HDF5_TRAIN
        shuffle = True
        transform = get_transform(cfg)
        load_aux_data = cfg.DATA.LOAD_AUXILIARY_DATA
    elif mode == "val":
        data_path = cfg.DATA.PATH_HDF5_VAL
        cfg.DATA.FRACTION = 1.0
        shuffle = False
        load_aux_data = False
        transform = transforms.Compose([ToTensor()])
    else:
        raise ValueError("Invalid mode, must be 'val' or 'train'.")

    logger.info(f"Loading {mode.capitalize()} data ...")
    dataset = dset.CerebDataset(data_path, cfg, transform, load_aux_data)

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        num_workers=cfg.DATA_LOADER.NUM_WORKERS,
        shuffle=shuffle,
        pin_memory=True,
    )
    return dataloader
