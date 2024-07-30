
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

from CCNet.data_loader import dataset as dset
from CCNet.data_loader.augmentation import ToTensor, ZeroPad2D, \
    RandomizeScaleFactor, SmartRandomCutout, RandomCutout, CutoutRandomHemisphere, FlipLeftRight, \
    CutoutBRATSTumor, CutoutBRATSTumorDeterministic, CutoutTumorMask, RandGridDistortiond
from FastSurferCNN.utils import logging
#from monai.utils import GridSampleMode, GridSamplePadMode

logger = logging.getLogger(__name__)





def create_all_augmentations(plane: str):
    """"
    create a dictionary of all augmentations
    """
    # Cutout
    cutout = RandomCutout(include=['image'], downweighting_factor=0.5)

    if plane is not None:
        hemisphere = CutoutRandomHemisphere(orientation=plane, include=['image'])
    else:
        hemisphere = None


    # brats_tumor = CutoutBRATSTumor(tumor_mask_hdf5='../../data/tumor_masks.hdf5',
    #                                include=['image', 'weight'], downweighting_factor=0.9, random=False)

    # Flip
    flip = FlipLeftRight(orientation=plane, include=['image', 'label', 'weight', 'unmodified_center_slice', 'aux_labels', 'cutout_mask'])
    
    # NOTE: Don't use this!! Its super slow
    # Elastic
    # elastic = tio.RandomElasticDeformation(num_control_points=7,
    #                                         max_displacement=(0, 30, 30),
    #                                         locked_borders=2,
    #                                         image_interpolation='linear',
    #                                         include=['image', 'label', 'weight', 'unmodified_center_slice'])

    # Grid Distortion
    grid_distortion = RandGridDistortiond(keys=['image', 'weight', 'label', 'unmodified_center_slice', 'cutout_mask'], 
                                            num_cells=20, prob=1, distort_limit=(-0.4, 0.4), mode='nearest', padding_mode='zeros') #mode=GridSampleMode.NEAREST, padding_mode=GridSamplePadMode.ZEROS),

    # Scales
    scaling = tio.RandomAffine(scales=(0.8, 1.15),
                                degrees=0,
                                translation=(0, 0, 0),
                                isotropic=True,  # If True, scaling factor along all dimensions is the same
                                center='image',
                                default_pad_value='minimum',
                                image_interpolation='linear',
                                include=['image', 'label', 'weight', 'unmodified_center_slice', 'aux_labels', 'cutout_mask'])

    # Rotation
    rot = tio.RandomAffine(scales=(1.0, 1.0),
                            degrees=(20,0,0),
                            translation=(0, 0, 0),
                            isotropic=True,  # If True, scaling factor along all dimensions is the same
                            center='image',
                            default_pad_value='minimum',
                            image_interpolation='linear',
                            include=['image', 'label', 'weight', 'unmodified_center_slice', 'aux_labels', 'cutout_mask'])

    # Translation
    tl = tio.RandomAffine(scales=(1.0, 1.0),
                            degrees=0,
                            translation=(15.0, 15.0, 0),
                            isotropic=True,  # If True, scaling factor along all dimensions is the same
                            center='image',
                            default_pad_value='minimum',
                            image_interpolation='linear',
                            include=['image', 'label', 'weight', 'unmodified_center_slice', 'aux_labels', 'cutout_mask'])

    # Customzied Affine with scaling and rotation for better performance
    custom_affine = tio.RandomAffine(scales=(0.8, 1.15),
                            degrees=(10,0,0),
                            translation=(0, 0, 0),
                            isotropic=True,  # If True, scaling factor along all dimensions is the same
                            center='image',
                            default_pad_value='minimum',
                            image_interpolation='linear',
                            include=['image', 'label', 'weight', 'unmodified_center_slice', 'aux_labels', 'cutout_mask'])

    # Random Anisotropy (Downsample image along an axis, then upsample back to initial space
    ra = tio.transforms.RandomAnisotropy(axes=(0, 1),
                                            downsampling=(1.1, 1.5),
                                            image_interpolation="linear",
                                            include=['image', 'unmodified_center_slice'])

    # Bias Field
    bias_field = tio.transforms.RandomBiasField(coefficients=0.5, order=3, include=['image', 'unmodified_center_slice'])

    # Gamma
    random_gamma = tio.transforms.RandomGamma(log_gamma=(-0.1, 0.1), include=['image', 'unmodified_center_slice'])

    return {"Cutout": cutout,
            "Hemisphere": hemisphere, "Flip": flip,
            "Scaling": scaling, "Rotation": rot, 
            "CustomAffine": custom_affine, "Translation": tl,
            "RAnisotropy": ra, "BiasField": bias_field, "RGamma": random_gamma, 
            "GridDistortion": grid_distortion}


def get_dataloader(cfg, mode, val_dataset=None, data_path=None):
    """
    Creating the dataset and pytorch data loader

    :param cfg:
    :param mode: loading data for train, val and test mode
    :return:
    """
    if mode == 'train':
        batch_size = cfg.TRAIN.BATCH_SIZE
        if "None" in cfg.DATA.AUG:
            tfs = [ToTensor()]#ZeroPad2D((padding_size, padding_size)), 
            # old transform
            if "Gaussian" in cfg.DATA.AUG:
                tfs.append(RandomizeScaleFactor(mean=0, std=0.1))

            if data_path is None:
                data_path = cfg.DATA.PATH_HDF5_TRAIN
            shuffle = True

            logger.info(f"Loading {mode.capitalize()} data ... from {data_path}. Using legacy data augmentation (no torchio)")

            dataset = dset.MultiScaleDatasetVal(data_path, cfg, transforms.Compose(tfs))
        elif len(cfg.DATA.AUG) == 0:
            if data_path is None:
                data_path = cfg.DATA.PATH_HDF5_TRAIN
            shuffle = True

            logger.info(f"Loading {mode.capitalize()} data from {data_path}. No data augmentation")

            dataset = dset.MultiScaleDataset(data_path, cfg, scale_aug=False, transforms=None)
        else:

            all_augmentations = create_all_augmentations(cfg.DATA.PLANE)

            all_tfs = {all_augmentations[aug] for aug in cfg.DATA.AUG if aug != "Gaussian"} # TODO: adjust hard coded probability (is this the same as hardcoded below?)
            gaussian_noise = True if "Gaussian" in cfg.DATA.AUG else False

            # remove "Gaussian" from cfg.DATA.AUG
            cfg.DATA.AUG = [aug for aug in cfg.DATA.AUG if aug != "Gaussian"]

            # get probabilities for each augmentation
            if cfg.DATA.AUG_LIKELYHOOD is None:
                cfg.DATA.AUG_LIKELYHOOD = 1.0
            if isinstance(cfg.DATA.AUG_LIKELYHOOD, float):
                cfg.DATA.AUG_LIKELYHOOD = [cfg.DATA.AUG_LIKELYHOOD] * len(cfg.DATA.AUG)

            for prob, aug in zip(cfg.DATA.AUG_LIKELYHOOD, all_tfs):
                aug.p = prob # set probability for each augmentation
            #all_tfs = [tio.OneOf(all_tfs, p=prob) for prob in cfg.DATA.AUG_LIKELYHOOD]

            transform = tio.Compose(all_tfs, include=["image", "label", "weight"])

            data_path = cfg.DATA.PATH_HDF5_TRAIN
            shuffle = True

            logger.info(f"Loading {mode.capitalize()} data from {data_path}. Using torchio data augmentation")

            dataset = dset.MultiScaleDataset(data_path, cfg, gaussian_noise, transforms=transform)

        

    elif mode.startswith('val'):
        if data_path is None:
            data_path = cfg.DATA.PATH_HDF5_VAL
        

        shuffle = False
        batch_size = cfg.TEST.BATCH_SIZE

        if mode.endswith('inpainting'):
            logger.info(f"Loading {mode.capitalize()} data from {data_path}. Using inpainting data augmentation")
            transform = CutoutBRATSTumorDeterministic(tumor_mask_hdf5='../../data/tumor_masks.hdf5', include=['image'])
            dataset = dset.MultiScaleDataset(data_path, cfg, scale_aug=False, transforms=transform)
        else: #mode == 'val_inpainting'
            logger.info(f"Loading {mode.capitalize()} data from {data_path}")
            transform = CutoutTumorMask()
            #dataset = dset.InpaintingDataset(val_dataset)
            dataset = dset.MultiScaleDataset(data_path, cfg, scale_aug=False, transforms=transform)

        

    else:
        raise ValueError(f"Unknown dataloader mode {mode}")
    
    assert(len(dataset) > 0), f"Dataset {data_path} is empty"


    if cfg.DATA_LOADER.NUM_WORKERS > 0:
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=cfg.DATA_LOADER.NUM_WORKERS,
            shuffle=shuffle,
            pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
            prefetch_factor=cfg.DATA_LOADER.PREFETCH_FACTOR, # prefetch doesn't do anything if we don't have workers
            drop_last=False
        )
    else:
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=cfg.DATA_LOADER.NUM_WORKERS,
            shuffle=shuffle,
            pin_memory=cfg.DATA_LOADER.PIN_MEMORY,  # prefetch doesn't do anything if we don't have workers
            drop_last=False
        )
    return dataloader
