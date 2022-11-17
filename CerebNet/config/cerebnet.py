from yacs.config import CfgNode as CN

_C = CN()

# ---------------------------------------------------------------------------- #
# Model options
# ---------------------------------------------------------------------------- #
_C.MODEL = CN()

# Name of model
_C.MODEL.MODEL_NAME = "FastSurferCNN"

# Number of classes to predict (axial and coronal), including background
_C.MODEL.NUM_CLASSES = 28

# Merged Number of classes for sagittal view
_C.MODEL.NUM_CLASSES_SAG = 17

# Loss function, combined = dice loss + cross entropy/ multi_loss
_C.MODEL.LOSS_FUNC = "combined"

# to use weight mask or not
_C.MODEL.WEIGHT_MASK = True

# Filter dimensions for DenseNet (all layers same)
_C.MODEL.NUM_FILTERS = 64

# Number of input channels (slice thickness)
_C.MODEL.NUM_CHANNELS = 7


_C.MODEL.DILATION = 1

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
_C.MODEL.HEIGHT = 128

# The width of segmentation model
_C.MODEL.WIDTH = 128


# ---------------------------------------------------------------------------- #
# Loss options
# ---------------------------------------------------------------------------- #
_C.LOSS = CN()

_C.LOSS.IGNORE_CLASSES = True


# ---------------------------------------------------------------------------- #
# Training options
# ---------------------------------------------------------------------------- #
_C.TRAIN = CN()

_C.TRAIN.ENABLE = True

# Development mode for training
_C.TRAIN.DEV_MODE = False


# DO an eval loop for debugging 
_C.TRAIN.EVAL = False

# input batch size for training
_C.TRAIN.BATCH_SIZE = 16

# how many batches to wait before logging training status
_C.TRAIN.LOG_INTERVAL = 50

# Resume training from the latest checkpoint in the output directory.
_C.TRAIN.RESUME = False

# The experiment number to resume from
_C.TRAIN.RESUME_EXPR_NUM = -1

# number of epochs to train
_C.TRAIN.NUM_EPOCHS = 30

# number of steps to train
_C.TRAIN.NUM_STEPS = 30

# To fine tune model or not
_C.TRAIN.FINE_TUNE = False

# Evaluate model on test data every eval period epochs.
_C.TRAIN.EVAL_PERIOD = 2

# checkpoint period
_C.TRAIN.CHECKPOINT_PERIOD = 2

# checkpoint to load model for fine tuning
_C.TRAIN.CHECKPOINT_PATH = ""

# dropping the last conv layer to fine tune
# with different number of classes
_C.TRAIN.DROP_LAST_CONV = False

# ---------------------------------------------------------------------------- #
# Data Augmentation options
# ---------------------------------------------------------------------------- #

# Augmentation for traning
_C.AUGMENTATION = CN()

# list of augmentations to use for training
# random_affine, flip, bias_field, warp
_C.AUGMENTATION.TYPES = []

# The probability to apply augmentation
_C.AUGMENTATION.PROB = 0.5

# random degree from interval [-DEGREE, DEGREE] for rotation
_C.AUGMENTATION.DEGREE = 20

# random scaling factor from SCALE interval
_C.AUGMENTATION.SCALE = (0.95, 1.15)

# random offset from interval (-TRANSLATE*IMG_SIZE, TRANSLATE*IMG_SIZE)
# for translation, TRANSLATE is a number between 0 and 1.0
_C.AUGMENTATION.TRANSLATE = 0.1

# random bias field coefficient range
# a tuple of (min_range,max_range) or single number a -> (-a, a)
_C.AUGMENTATION.BIAS_FIELD_COEFFICIENTS = (-0.5, 0.5)

# order bias field
_C.AUGMENTATION.BIAS_FIELD_ORDER = 3

# The axis to flip image along
# 160 cases: coronal: 0, axial:1
# dzne_manual_native: 0 for both coronal and axial
_C.AUGMENTATION.FLIP_AXIS = 0


# ---------------------------------------------------------------------------- #
# Testing options
# ---------------------------------------------------------------------------- #
_C.TEST = CN()

# Enable full eval with view-aggregation
_C.TEST.ENABLE = False

# path to config file to load subject data
# which provides the options on how to  load subject's data
_C.TEST.DATA_CONFIG_FILE = ""

# input batch size for eval
_C.TEST.BATCH_SIZE = 4

# to use gpu or not
_C.TEST.USE_CUDA = True

# the folder to store the output mri images and metrics
_C.TEST.RESULTS_DIR = "results"

# the path to checkpoint for each view which includes the expr num
_C.TEST.AXIAL_CHECKPOINT_PATH = ""

_C.TEST.CORONAL_CHECKPOINT_PATH = ""

_C.TEST.SAGITTAL_CHECKPOINT_PATH = ""

# name of the prediction image to be saved
_C.TEST.PREDICTION_OUT_FILENAME = "CerebNet_Pred.nii.gz"

# path to folder of subjects
_C.TEST.DATA_PATH = ''

# full path to csv file with list of subjects
_C.TEST.SUBJECT_CSV_PATH = ''

# split number in cross-validation
_C.TEST.SPLIT_NUM = 1


# ---------------------------------------------------------------------------- #
# Data options
# ---------------------------------------------------------------------------- #


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

# using fraction of auxiliary data
_C.DATA.FRACTION = 1.0

# number of cross-validation splits
_C.DATA.SPLIT_NUM = 1

# to use talairach coordinates for training or not
_C.DATA.LOAD_TALAIRACH = False

# Number of pre- and succeeding slices (default: 3), 0 means no thickness
_C.DATA.THICKNESS = 3

# Patch sizes used for network to process
_C.DATA.PATCH_SIZE = (128, 128, 128)

_C.DATA.EDGE_WEIGHING = True

# the direction of primary slice,
# the plane (axial, coronal, sagittal) that the last dimension of
# 3D volume corresponds to
_C.DATA.PRIMARY_SLICE_DIR = "axial"

_C.DATA.LOAD_AUXILIARY_DATA = False

# ---------------------------------------------------------------------------- #
# DataLoader options (common for test and train)
# ---------------------------------------------------------------------------- #
_C.DATA_LOADER = CN()

# Number of data loader workers
_C.DATA_LOADER.NUM_WORKERS = 0

# Load data to pinned host memory.
_C.DATA_LOADER.PIN_MEMORY = True

# ---------------------------------------------------------------------------- #
# Optimizer options
# ---------------------------------------------------------------------------- #
_C.OPTIMIZER = CN()

# Base learning rate.
_C.OPTIMIZER.BASE_LR = 0.01

# Learning rate scheduler, step_lr, cosineWarmRestarts, multiStep
_C.OPTIMIZER.LR_SCHEDULER = "multiStep"

# Multiplicative factor of learning rate decay
_C.OPTIMIZER.GAMMA = 0.1

# minimum learning in cosineWarmRestarts and reduceOnPlateau(min_lr)
_C.OPTIMIZER.ETA_MIN = 1e-4

# StepLR params -----------------------------
# Period of learning rate decay in step_lr
_C.OPTIMIZER.STEP_SIZE = 5

# ReduceOnPlateau params --------------------------
_C.OPTIMIZER.PATIENCE = 4

# Cosine-based lr scheduler params --------------------------
# Number of iterations for the first restart in cosineWarmRestarts
_C.OPTIMIZER.T_ZERO = 10
# A factor increases T_i after a restart in cosineWarmRestarts
_C.OPTIMIZER.T_MULT = 2

# MultiStep lr scheduler params -----------------------------
_C.OPTIMIZER.MILESTONES = [20, 40]

# if the scheduler is metric based and should step after eval
_C.OPTIMIZER.METRIC_SCHEDULER = False

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

_C.DEBUG = False

# Note that non-determinism may still be present due to non-deterministic
# operator implementations in GPU operator libraries.
_C.RNG_SEED = 1


def get_cfg_cerebnet() -> CN:
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()
