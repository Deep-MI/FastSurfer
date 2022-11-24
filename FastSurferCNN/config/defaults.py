from yacs.config import CfgNode as CN

_C = CN()

# ---------------------------------------------------------------------------- #
# Model options
# ---------------------------------------------------------------------------- #
_C.MODEL = CN()

# Name of model
_C.MODEL.MODEL_NAME = "FastSurferVINN"

# Number of classes to predict, including background
_C.MODEL.NUM_CLASSES = 79

# Loss function, combined = dice loss + cross entropy, combined2 = dice loss + boundary loss
_C.MODEL.LOSS_FUNC = "combined"

# Filter dimensions for DenseNet (all layers same)
_C.MODEL.NUM_FILTERS = 71

# Filter dimensions for Input Interpolation block (currently all the same)
_C.MODEL.NUM_FILTERS_INTERPOL = 32

# Number of UNet layers in Basenetwork (including bottleneck layer!)
_C.MODEL.NUM_BLOCKS = 5

# Number of input channels (slice thickness)
_C.MODEL.NUM_CHANNELS = 7

# Height of convolution kernels
_C.MODEL.KERNEL_H = 3

# Width of convolution kernels
_C.MODEL.KERNEL_W = 3

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
_C.TRAIN.RESUME_EXPR_NUM = "Default"

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

# Flag to disable or enable Early Stopping
_C.TRAIN.EARLY_STOPPING = True

# Mode for early stopping (min = stop when metric is no longer decreasing, max = stop when mwtric is no longer increasing)
_C.TRAIN.EARLY_STOPPING_MODE = "min"

# Patience = Number of epochs to wait before stopping
_C.TRAIN.EARLY_STOPPING_PATIENCE = 10

# Wait = NUmber of epochs before starting early stopping check
_C.TRAIN.EARLY_STOPPING_WAIT = 10

# Delta = change below which early stopping starts (previous - current < delta = stop)
_C.TRAIN.EARLY_STOPPING_DELTA = 0.00001

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

# Which classes to use
_C.DATA.CLASS_OPTIONS = ["aseg", "aparc"]

# Available size for dataloader
# This for the multi-scale dataloader
_C.DATA.SIZES = [256, 311, 320]

# the size that all inputs are padded to
_C.DATA.PADDED_SIZE = 320

# Augmentations
_C.DATA.AUG = ["Scaling", "Translation"]

# ---------------------------------------------------------------------------- #
# DataLoader options (common for test and train)
# ---------------------------------------------------------------------------- #
_C.DATA_LOADER = CN()

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

# Learning rate scheduler, step_lr, cosineWarmRestarts, reduceLROnPlateau
_C.OPTIMIZER.LR_SCHEDULER = "cosineWarmRestarts"

# Multiplicative factor of learning rate decay in step_lr
_C.OPTIMIZER.GAMMA = 0.3

# Period of learning rate decay in step_lr
_C.OPTIMIZER.STEP_SIZE = 5

# minimum learning in cosine lr policy and reduceLROnPlateau
_C.OPTIMIZER.ETA_MIN = 0.0001

# number of iterations for the first restart in cosineWarmRestarts
_C.OPTIMIZER.T_ZERO = 10

# A factor increases T_i after a restart in cosineWarmRestarts
_C.OPTIMIZER.T_MULT = 2

# factor by which learning rate will be reduce (new_lr = lr*factor, default=0.1)
_C.OPTIMIZER.FACTOR = 0.1

# number of epochs to wait before lowering lr (default=5)
_C.OPTIMIZER.PATIENCE = 5

# Threshold for measuring new optimum (default=1e-4)
_C.OPTIMIZER.THRESH = 0.0001

# Number of epochs to wait before resuming normal operation (default=0)
_C.OPTIMIZER.COOLDOWN=0

# Momentum
_C.OPTIMIZER.MOMENTUM = 0.9

# Momentum dampening
_C.OPTIMIZER.DAMPENING = 0.0

# Nesterov momentum
_C.OPTIMIZER.NESTEROV = True

# L2 regularization
_C.OPTIMIZER.WEIGHT_DECAY = 1e-4

# Optimization method [sgd, adam, adamW]
_C.OPTIMIZER.OPTIMIZING_METHOD = "adamW"

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


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()