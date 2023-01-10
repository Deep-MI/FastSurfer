# Overview

This directory contains all information needed to run inference with the readily trained FastSurferVINN or train it from scratch. FastSurferCNN is capable of whole brain segmentation into 95 classes in under 1 minute, mimicking FreeSurfer's anatomical segmentation and cortical parcellation (DKTatlas). The network architecture incorporates local and global competition via competitive dense blocks and competitive skip pathways, as well as multi-slice information aggregation that specifically tailor network performance towards accurate segmentation of both cortical and sub-cortical structures. 
![](/images/detailed_network.png)

The network was trained with conformed images (UCHAR, 1-0.7 mm voxels and standard slice orientation). These specifications are checked in the run_prediction.py script and the image is automatically conformed if it does not comply.

# 1. Inference

The *FastSurferCNN* directory contains all the source code and modules needed to run the scripts. A list of python libraries used within the code can be found in __requirements.txt__. The main script is called __run_prediction.py__ within which certain options can be selected and set via the command line:

#### General
* `--in_dir`: Path to the input volume directory (e.g /your/path/to/ADNI/fs60) or 
* `--csv_file`: Path to csv-file listing input volume directories
* `--t1`: name of the T1-weighted MRI_volume (like mri_volume.mgz, __default: orig.mgz__)
* `--conformed_name`: name of the conformed MRI_volume (the input volume is always first conformed, if not already, and the result is saved under the given name, __default: orig.mgz__)
* `--t`: search tag limits processing to subjects matching the pattern (e.g. sub-* or 1030*...)
* `--sd`: Path to output directory (where should predictions be saved). Will be created if it does not already exist.
* `--seg_log`: name of log-file (information about processing is stored here; If not set, logs will not be saved). Saved in the same directory as the predictions.
* `--strip`: strip suffix from path definition of input file to yield correct subject name. (Optional, if full path is defined for `--t1`)
* `--lut`: FreeSurfer-style Color Lookup Table with labels to use in final prediction. Default: ./config/FastSurfer_ColorLUT.tsv
* `--seg`: Name of intermediate DL-based segmentation file (similar to aparc+aseg).
* `--cfg_cor`: Path to the coronal config file
* `--cfg_sag`: Path to the axial config file
* `--cfg_ax`: Path to the sagittal config file

#### Checkpoints
* `--ckpt_sag`: path to sagittal network checkpoint
* `--ckpt_cor`: path to coronal network checkpoint
* `--ckpt_ax`: path to axial network checkpoint

#### Optional commands
* `--clean`: clean up segmentation after running it (optional)
* `--device <str>`:Device for processing (_auto_, _cpu_, _cuda_, _cuda:<device_num>_), where cuda means Nvidia GPU; you can select which one e.g. "cuda:1". Default: "auto", check GPU and then CPU
* `--viewagg_device <str>`: Define where the view aggregation should be run on. 
                    Can be _auto_ or a device (see --device).
                    By default (_auto_), the program checks if you have enough memory to run the view aggregation on the gpu. 
                    The total memory is considered for this decision. 
                    If this fails, or you actively overwrote the check with setting `--viewagg_device cpu`, view agg is run on the cpu. 
                    Equivalently, if you define `--viewagg_device gpu`, view agg will be run on the gpu (no memory check will be done).
* `--batch_size`: Batch size for inference. Default=1


### Example Command Evaluation Single Subject
To run the network on MRI-volumes of subjectX in ./data (specified by `--t1` flag; e.g. ./data/subjectX/t1-weighted.nii.gz), change into the *FastSurferCNN* directory and run the following commands: 

```
python3 run_prediction.py --t1 ../data/subjectX/t1-weighted.nii.gz \
--sd ../output \
--t subjectX \
--seg_log ../output/temp_Competitive.log \
```

The output will be stored in:

- ../output/subjectX/mri/aparc.DKTatlas+aseg.deep.mgz (large segmentation)
- ../output/subjectX/mri/mask.mgz (brain mask)
- ../output/subjectX/mri/aseg_noCC.mgz (reduced segmentation)

Here the logfile "temp_Competitive.log" will include the logfiles of all subjects. If left out, the logs will be written to stdout


### Example Command Evaluation whole directory
To run the network on all subjects MRI-volumes in ./data, change into the *FastSurferCNN* directory and run the following command: 

```
python3 run_prediction.py --in_dir ../data \
--sd ../output \
--seg_log ../output/temp_Competitive.log \
```

The output will be stored in:

- ../output/subjectX/mri/aparc.DKTatlas+aseg.deep.mgz (large segmentation)
- ../output/subjectX/mri/mask.mgz (brain mask)
- ../output/subjectX/mri/aseg_noCC.mgz (reduced segmentation)
- and the log in ../output/temp_Competitive.log


# 2. Hdf5-Trainingset Generation

The *FastSurferCNN* directory contains all the source code and modules needed to create a hdf5-file from given MRI volumes. Here, we use the orig.mgz output from freesurfer as the input image and the aparc.DKTatlas+aseg.mgz as the ground truth. The mapping functions are set-up accordingly as well and need to be changed if you use a different segmentation as ground truth. 
A list of python libraries used within the code can be found in __requirements.txt__. The main script is called __generate_hdf5.py__ within which certain options can be selected and set via the command line:

#### General
* `--hdf5_name`: Path and name of the to-be-created hdf5-file. Default: ../data/hdf5_set/Multires_coronal.hdf5
* `--data_dir`: Directory with images to load. Default: /data
* `--pattern`: Pattern to match only certain files in the directory
* `--csv_file`: Csv-file listing subjects to load (can be used instead of data_dir; one complete path per line (up to the subject directory))
              Example: You have a directory called **dataset** with three different datasets (**D1**, **D2** and **D3**). You want to include subject1, subject10 and subject20 from D1 and D2. Your csv-file would then look like this:
              
              /dataset/D1/subject1
              /dataset/D1/subject10
              /dataset/D1/subject20
              /dataset/D2/subject1
              /dataset/D2/subject10
              /dataset/D2/subject20
* --lut: FreeSurfer-style Color Lookup Table with labels to use in final prediction. Default: ./config/FastSurfer_ColorLUT.tsv
              
The actual filename and segmentation ground truth name is specified via `--image_name` and `--gt_name` (e.g. the actual file could be sth. like /dataset/D1/subject1/mri_volume.mgz and /dataset/D1/subject1/segmentation.mgz)

#### Image Names
* `--image_name`: Default name of original images. FreeSurfer orig.mgz is default (mri/orig.mgz)
* `--gt_name`: Default name for ground truth segmentations. Default: mri/aparc.DKTatlas+aseg.mgz.
* `--gt_nocc`: Segmentation without corpus callosum (used to mask this segmentation in ground truth). For a normal FreeSurfer input, use mri/aseg.auto_noCCseg.mgz. 

#### Image specific options
* `--plane`: Which anatomical plane to use for slicing (axial, coronal or sagittal)
* `--thickness`: Number of pre- and succeeding slices (we use 3 --> total of 7 slices is fed to the network; default: 3)
* `--combi`: Suffixes of labels names to combine. Default: Left- and Right-
* `--sag_mask`: Suffixes of labels names to mask for final sagittal labels. Default: Left- and ctx-rh
* `--max_w`: Overall max weight for any voxel in weight mask. Default: 5
* `--hires_w`: Weight for high resolution elements (sulci, WM strands, cortex border) in weight mask. Default: None
* `--no_grad`: Turn on to only use median weight frequency (no gradient). Default: False
* `--gm`: Turn on to add cortex mask for hires-processing. Default: False
* `--processing`: Use aseg, aparc or no specific mapping processing. Default: aparc
* `--sizes`: Resolutions of images in the dataset. Default: 256
* `--edge_w`: Weight for edges in weight mask. Default=5

#### Example Command Axial (Single Resolution)
```
python3 generate_hdf5.py \
--hdf5_name ../data/training_set_axial.hdf5 \
--csv_file ../training_set_subjects_dirs.csv \
--thickness 3 \
--plane axial \
--image_name mri/orig.mgz \
--gt_name mri/aparc.DKTatlas+aseg.mgz \
--gt_nocc mri/aseg.auto_noCCseg.mgz
--max_w 5 \
--edge_w 4 \
--hires_w 4 \
--sizes 256

```

#### Example Command Coronal (Single Resolution)
```
python3 generate_hdf5.py \
--hdf5_name ../data/training_set_coronal.hdf5 \
--csv_file ../training_set_subjects_dirs.csv \
--plane coronal \
--image_name mri/orig.mgz \
--gt_name mri/aparc.DKTatlas+aseg.mgz \
--gt_nocc mri/aseg.auto_noCCseg.mgz
--max_w 5 \
--edge_w 4 \
--hires_w 4 \
--sizes 256

```

#### Example Command Sagittal (Multiple Resolutions)
```
python3 generate_hdf5.py \
--hdf5_name ../data/training_set_sagittal.hdf5 \
--csv_file ../training_set_subjects_dirs.csv \
--plane sagittal \
--image_name mri/orig.mgz \
--gt_name mri/aparc.DKTatlas+aseg.mgz \
--gt_nocc mri/aseg.auto_noCCseg.mgz
--max_w 5 \
--edge_w 4 \
--hires_w 4 \
--sizes 256 311 320

```

#### Example Command Sagittal using --data_dir instead of --csv_file
`--data_dir` specifies the path in which the data is located, with `--pattern` we can select subjects from the specified path. By default the pattern is "*" meaning all subjects will be selected.
As an example, imagine you have 19 FreeSurfer processed subjects labeled subject1 to subject19 in the ../data directory:

```
/home/user/FastSurfer/data
├── subject1
├── subject2
├── subject3
…
│
├── subject19
    ├── mri
    │   ├── aparc.DKTatlas+aseg.mgz
    │   ├── aseg.auto_noCCseg.mgz
    │   ├── orig.mgz
    │   ├── …
    │   …
    ├── scripts
    ├── stats
    ├── surf
    ├── tmp
    ├── touch
    └── trash
```

Setting `--pattern` "*" will select all 19 subjects (subject1, ..., subject19).
Now, if only a subset should be used for the hdf5-file (e.g. subject 10 till subject19), this can be done by changing the `--pattern` flag to "subject1[0-9]": 

```
python3 generate_hdf5.py \
--hdf5_name ../data/training_set_axial.hdf5 \
--data_dir ../data \
--pattern "subject1[0-9]" \
--plane sagittal \
--image_name mri/orig.mgz \
--gt_name mri/aparc.DKTatlas+aseg.mgz \
--gt_nocc mri/aseg.auto_noCCseg.mgz
 
```

# 3. Training

The *FastSurferCNN* directory contains all the source code and modules needed to run the scripts. A list of python libraries used within the code can be found in __requirements.txt__. The main training script is called __run_model.py__ whose options can be set through a configuration file and command line arguments:
* `--cfg`: Path to the configuration file. Default: config/FastSurferVINN.yaml
* `--aug`: List of augmentations to use. Default: None.
* `--opt`: List of class options to use.

The `--cfg` file configures the model to be trained. See config/FastSurferVINN.yaml for an example and config/defaults.py for all options and default values.

The configuration options include:

#### Model options
* MODEL_NAME: Name of model [FastSurferCNN, FastSurferVINN]. Default: FastSurferVINN
* NUM_CLASSES: Number of classes to predict including background. Axial and coronal: 79 (default), Sagittal: 51.
* NUM_FILTERS: Filter dimensions for Networks (all layers same). Default: 71
* NUM_CHANNELS: Number of input channels (slice thickness). Default: 7
* KERNEL_H: Height of Kernel. Default: 3
* KERNEL_W: Width of Kernel. Default: 3
* STRIDE_CONV: Stride during convolution. Default: 1
* STRIDE_POOL: Stride during pooling. Default: 2
* POOL: Size of pooling filter. Default: 2
* BASE_RES: Base resolution of the segmentation model (after interpolation layer). Default: 1

#### Optimizer options

* BASE_LR: Base learning rate. Default: 0.01
* OPTIMIZING_METHOD: Optimization method [sgd, adam, adamW]. Default: adamW
* MOMENTUM: Momentum for optimizer. Default: 0.9
* NESTEROV: Enables Nesterov for optimizer. Default: True
* LR_SCHEDULER: Learning rate scheduler [step_lr, cosineWarmRestarts, reduceLROnPlateau]. Default: cosineWarmRestarts


#### Data options

* PATH_HDF5_TRAIN: Path to training hdf5-dataset
* PATH_HDF5_VAL: Path to validation hdf5-dataset
* PLANE: Plane to load [axial, coronal, sagittal]. Default: coronal

#### Training options

* BATCH_SIZE: Input batch size for training. Default: 16
* NUM_EPOCHS: Number of epochs to train. Default: 30
* SIZES: Available image sizes for the multi-scale dataloader. Default: [256, 311 and 320]
* AUG: Augmentations. Default: ["Scaling", "Translation"]

#### Misc. Options

* LOG_DIR: Log directory for run
* NUM_GPUS: Number of GPUs to use. Default: 1
* RNG_SEED: Select random seed. Default: 1


Any option can alternatively be set through the command-line by specifying the option name (as defined in config/defaults.py) followed by a value, such as: `MODEL.NUM_CLASSES 51`.

To train the network on a given hdf5-set, change into the *FastSurferCNN* directory and run
`run_model.py` as in the following examples:

### Example Command: Training Default FastSurferVINN
Trains FastSurferVINN on multi-resolution images in the coronal plane:
```
python3 run_model.py \
--cfg ./config/FastSurferVINN.yaml
```

### Example Command: Training FastSurferVINN (Single Resolution)
Trains FastSurferVINN on single-resolution images in the sagittal plane by overriding the NUM_CLASSES, SIZES, PATH_HDF5_TRAIN, and PATH_HDF5_VAL options:
```
python3 run_model.py \
--cfg ./config/FastSurferVINN.yaml \
MODEL.NUM_CLASSES 51 \
DATA.SIZES 256 \
DATA.PATH_HDF5_TRAIN ./hdf5_sets/training_sagittal_single_resolution.hdf5 \
DATA.PATH_HDF5_VAL ./hdf5_sets/validation_sagittal_single_resolution.hdf5 \
```

### Example Command: Training FastSurferCNN
Trains FastSurferCNN using a provided configuration file and specifying no augmentations:
```
python3 run_model.py \
--cfg custom_configs/FastSurferCNN.yaml \
--aug None
```
