# Copyright 2024 AI in Medical Imaging, German Center for Neurodegenerative Diseases(DZNE), Bonn
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

from yacs.config import CfgNode as CN

_C = CN()

# ---------------------------------------------------------------------------- #
# Model options
# ---------------------------------------------------------------------------- #
_C.MODEL = CN()

# Name of model
_C.MODEL.MODEL_NAME = ""

#modalities 't1', 't2' or multi
_C.MODEL.MODE ='t1'

# Number of classes to predict, including background
_C.MODEL.NUM_CLASSES = 79

# Loss function, combined = dice loss + cross entropy, combined2 = dice loss + boundary loss
_C.MODEL.LOSS_FUNC = "combined"

# Filter dimensions for DenseNet (all layers same)
_C.MODEL.NUM_FILTERS = 64

# Filter dimensions for Input Interpolation block (currently all the same)
_C.MODEL.NUM_FILTERS_INTERPOL = 32

# Number of input channels (slice thickness)
_C.MODEL.NUM_CHANNELS = 7

# Number of branches for attention mechanism
_C.MODEL.NUM_BRANCHES = 5

# Height of convolution kernels
_C.MODEL.KERNEL_H = 5

# Width of convolution kernels
_C.MODEL.KERNEL_W = 5

# size of Classifier kernel
_C.MODEL.KERNEL_C = 1

# Stride during convolution
_C.MODEL.STRIDE_CONV = 1

# Stride during pooling
_C.MODEL.STRIDE_POOL = 2

# Size of pooling filter
_C.MODEL.POOL = 2

# The height of segmentation model (after interpolation layer)
_C.MODEL.HEIGHT = 256

# The width of segmentation model
_C.MODEL.WIDTH = 256

# The base resolution of the segmentation model (after interpolation layer)
_C.MODEL.BASE_RES = 1.0

# Interpolation mode for up/downsampling in Flex networks
_C.MODEL.INTERPOLATION_MODE = "bilinear"

# Crop positions for up/downsampling in Flex networks
_C.MODEL.CROP_POSITION = "top_left"

# Out Tensor dimensions for interpolation layer
_C.MODEL.OUT_TENSOR_WIDTH = 320
_C.MODEL.OUT_TENSOR_HEIGHT = 320

# Flag, for smoothing testing (double number of feature maps before/after interpolation block)
_C.MODEL.SMOOTH = False

# Options for attention
_C.MODEL.ATTENTION_BASE = False
_C.MODEL.ATTENTION_INPUT = False
_C.MODEL.ATTENTION_OUTPUT = False

# Options for addition instead of Maxout
_C.MODEL.ADDITION = False

#Options for multi modalitie
_C.MODEL.MULTI_AUTO_W = False # weight per modalitiy
_C.MODEL.MULTI_AUTO_W_CHANNELS = False #weight per channel
# Flag, for smoothing testing (double number of feature maps before the input interpolation block)
_C.MODEL.MULTI_SMOOTH = False
#  Brach weights can be aleatory set to zero
_C.MODEL.HETERO_INPUT = False
# Flag for replicating any given modality into the two branches. This branch require that the hetero_input also set to TRUE
_C.MODEL.DUPLICATE_INPUT = False
# ---------------------------------------------------------------------------- #
# Training options
# ---------------------------------------------------------------------------- #
_C.TRAIN = CN()

# input batch size for training
_C.TRAIN.BATCH_SIZE = 16

# how many batches to wait before logging training status
_C.TRAIN.LOG_INTERVAL = 50

# Resume training from the latest checkpoint in the output directory.
_C.TRAIN.RESUME = False

# The experiment number to resume from
_C.TRAIN.RESUME_EXPR_NUM = 1

# number of epochs to train
_C.TRAIN.NUM_EPOCHS = 30

# number of steps (iteration) which depends on dataset
_C.TRAIN.NUM_STEPS = 10

# To fine tune model or not
_C.TRAIN.FINE_TUNE = False

# checkpoint period
_C.TRAIN.CHECKPOINT_PERIOD = 2

# number of worker for dataloader
_C.TRAIN.NUM_WORKERS = 8

# run validation
_C.TRAIN.RUN_VAL = True

# ---------------------------------------------------------------------------- #
# Testing options
# ---------------------------------------------------------------------------- #
_C.TEST = CN()

# input batch size for testing
_C.TEST.BATCH_SIZE = 16

# ---------------------------------------------------------------------------- #
# Data options
# ---------------------------------------------------------------------------- #

_C.DATA = CN()

# path to training hdf5-dataset
_C.DATA.PATH_HDF5_TRAIN = ""

# path to validation hdf5-dataset
_C.DATA.PATH_HDF5_VAL = ""

# The plane to load ['axial', 'coronal', 'sagittal']
_C.DATA.PLANE = "coronal"

# Reference volume frame during training, -1 value randomly select the frame : input data B,H,W,C,FRAME
_C.DATA.REF_FRAME = 0

# Reference volume frame during validation : input data B,H,W,C,FRAME
_C.DATA.VAL_REF_FRAME = 0

# Available size for dataloader
# This for the multi-scale dataloader
_C.DATA.SIZES = [256, 311, 320]

# the size that all inputs are padded to
_C.DATA.PADDED_SIZE = 320

# classes to consider in the Boundary loss (default: all -> 79)
_C.DATA.BOUNDARY_CLASSES = "None"

# Augmentations
_C.DATA.AUG = ["Flip", "Elastic", "Scaling", "Rotation", "Translation", "RAnisotropy", "BiasField", "RGamma"]

#Frequency of the hetero augmentations [both t1 and t2,only t1, only t2 ]
_C.DATA.HETERO_FREQ = [0.5, 0.25, 0.25]

# ---------------------------------------------------------------------------- #
# DataLoader options (common for test and train)
# ---------------------------------------------------------------------------- #
_C.DATA_LOADER = CN()

# the split number in cross validation
_C.DATA.SPLIT_NUM = 1
# Number of data loader workers
_C.DATA_LOADER.NUM_WORKERS = 8

# Load data to pinned host memory.
_C.DATA_LOADER.PIN_MEMORY = True

# ---------------------------------------------------------------------------- #
# Optimizer options
# ---------------------------------------------------------------------------- #
_C.OPTIMIZER = CN()

# Base learning rate.
_C.OPTIMIZER.BASE_LR = 0.01

# Learning rate scheduler, step_lr, cosineWarmRestarts
_C.OPTIMIZER.LR_SCHEDULER = "step_lr"

# Multiplicative factor of learning rate decay in step_lr
_C.OPTIMIZER.GAMMA = 0.3

# Period of learning rate decay in step_lr
_C.OPTIMIZER.STEP_SIZE = 5

# minimum learning in cosine lr policy
_C.OPTIMIZER.ETA_MIN = 0.0001

# number of iterations for the first restart in cosineWarmRestarts
_C.OPTIMIZER.T_ZERO = 10

# A factor increases T_i after a restart in cosineWarmRestarts
_C.OPTIMIZER.T_MULT = 2

# MultiStep lr scheduler params -----------------------------
_C.OPTIMIZER.MILESTONES = [20, 40]

# Momentum
_C.OPTIMIZER.MOMENTUM = 0.9

# Momentum dampening
_C.OPTIMIZER.DAMPENING = 0.0

# Nesterov momentum
_C.OPTIMIZER.NESTEROV = True

# L2 regularization
_C.OPTIMIZER.WEIGHT_DECAY = 1e-4

# Optimization method [sgd, adam]
_C.OPTIMIZER.OPTIMIZING_METHOD = "adam"

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #

# Number of GPUs to use
_C.NUM_GPUS = 1

# log directory for run
_C.LOG_DIR = "./experiments"

# experiment number
_C.EXPR_NUM = "Default"

# Note that non-determinism may still be present due to non-deterministic
# operator implementations in GPU operator libraries.
_C.RNG_SEED = 1


def get_cfg_hypvinn():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()