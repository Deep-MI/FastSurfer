# Overview

This directory contains all information needed to run inference with the readily trained FastSurferCNN or train it from scratch. FastSurferCNN is capable of whole brain segmentation into 95 classes in under 1 minute, mimicking FreeSurfer's anatomical segmentation and cortical parcellation (DKTatlas). The network architecture incorporates local and global competition via competitive dense blocks and competitive skip pathways, as well as multi-slice information aggregation that specifically tailor network performance towards accurate segmentation of both cortical and sub-cortical structures. 
![](/images/FastSurfer_v5.png)

The network was trained with conformed images (UCHAR, 256x256x256, 1 mm voxels and standard slice orientation). These specifications are checked in the eval.py script and the image is automatically conformed if it does not comply.

# 1. Inference

The *FastSurferCNN* directory contains all the source code and modules needed to run the scripts. A list of python libraries used within the code can be found in __requirements.txt__. The main script is called __eval.py__ within which certain options can be selected and set via the command line:

#### General
* --i_dir: Path to the input volume directory (e.g /your/path/to/ADNI/fs60) or 
* --csv_file: Path to csv-file listing input volume directories
* --in_name: name of the MRI_volume (like mri_volume.mgz, __default: orig.mgz__)
* --conformed_name: name of the conformed MRI_volume (the input volume is always first conformed, if not already, and the result is saved under the given name, __default: orig.mgz__)
* --t: search tag limits processing to subjects matching the pattern (e.g. sub-* or 1030*...)
* --o_dir: Path to output directory (where should predictions be saved). Will be created if it does not already exist.
* --out_name: name of the prediction; the extension determines the output file format, which can be one of: mgz, nii, or nii.gz (__default: aparc.DKTatlas+aseg.deep.mgz__)
* --log: name of log-file (information about processing is stored here; __default: deep-seg.log__). Saved in the same directory as the predictions.
* --order: order of interpolation (0=nearest,__1=linear(default)__, 2=quadratic, 3=cubic) for conformation (if not already done).

#### Checkpoints
* --network_sagittal_path: path to sagittal network checkpoint
* --network_coronal_path: path to coronal network checkpoint
* --network_axial_path: path to axial network checkpoint

#### Optional commands
* --clean: clean up segmentation after running it (optional)
* --run_viewagg_on: Define where the view aggregation should be run on. 
                    By default, the program checks if you have enough memory to run the view aggregation on the gpu. 
                    The total memory is considered for this decision. 
                    If this fails, or you actively overwrote the check with setting "--run_viewagg_on cpu", view agg is run on the cpu. 
                    Equivalently, if you define "--run_viewagg_on gpu", view agg will be run on the gpu (no memory check will be done).
* --no_cuda: Disable CUDA training (optional)


### Example Command Evaluation Single Subject
To run the network on MRI-volumes of subject1 in ./data (specified by --i_dir flag; e.g. ./data/subject1/orig.mgz), change into the *FastSurferCNN* directory and run the following commands: 

```
python3 eval.py --i_dir ../data \
--o_dir ../data \
--t subject1 \
--log temp_Competitive.log \
--network_sagittal_path ../checkpoints/Sagittal_Weights_FastSurferCNN/ckpts/Epoch_30_training_state.pkl \
--network_coronal_path ../checkpoints/Coronal_Weights_FastSurferCNN/ckpts/Epoch_30_training_state.pkl \
--network_axial_path ../checkpoints/Axial_Weights_FastSurferCNN/ckpts/Epoch_30_training_state.pkl 
```

The output will be saved in ./data/subject1/aparc.DKTatlas+aseg.deep.mgz.

### Example Command Evaluation whole directory
To run the network on all subjects MRI-volumes in ./data, change into the *FastSurferCNN* directory and run the following command: 

```
python3 eval.py --i_dir ../data \
--o_dir ../data \
--log temp_Competitive.log \
--network_sagittal_path ../checkpoints/Sagittal_Weights_FastSurferCNN/ckpts/Epoch_30_training_state.pkl \
--network_coronal_path ../checkpoints/Coronal_Weights_FastSurferCNN/ckpts/Epoch_30_training_state.pkl \
--network_axial_path ../checkpoints/Axial_Weights_FastSurferCNN/ckpts/Epoch_30_training_state.pkl 
```

The output will be stored in ./data/subjectX/aparc.DKTatlas+aseg.deep.mgz.

# 2. Hdf5-Trainingset Generation

The *FastSurferCNN* directory contains all the source code and modules needed to create a hdf5-file from given MRI volumes. Here, we use the orig.mgz output from freesurfer as the input image and the aparc.DKTatlas+aseg.mgz as the ground truth. The mapping functions are set-up accordingly as well and need to be changed if you use a different segmentation as ground truth. 
A list of python libraries used within the code can be found in __requirements.txt__. The main script is called __generate_hdf5.py__ within which certain options can be selected and set via the command line:

#### General
* --hdf5_name: Path and name of the to-be-created hdf5-file
* --data_dir: Directory with images to load
* --pattern: Pattern to match only certain files in the directory
* --csv_file: Csv-file listing subjects to load (can be used instead of data_dir; one complete path per line (up to the subject directory))
              Example: You have a directory called **dataset** with three different datasets (**D1**, **D2** and **D3**). You want to include subject1, subject10 and subject20 from D1 and D2. Your csv-file would then look like this:
              
              /dataset/D1/subject1
              /dataset/D1/subject10
              /dataset/D1/subject20
              /dataset/D2/subject1
              /dataset/D2/subject10
              /dataset/D2/subject20
              
The actual filename and segmentation ground truth name is specfied via --image_name and --gt_name (e.g. the actual file could be sth. like /dataset/D1/subject1/mri_volume.mgz and /dataset/D1/subject1/segmentation.mgz)

#### Image Names
* --image_name: Default name of original images. FreeSurfer orig.mgz is default (mri/orig.mgz)
* --gt_name: Default name for ground truth segmentations. Default: mri/aparc.DKTatlas+aseg.mgz.
* --gt_nocc: Segmentation without corpus callosum (used to mask this segmentation in ground truth). For a normal FreeSurfer input, use mri/aseg.auto_noCCseg.mgz. 

#### Image specific options
* --plane: which anatomical plane to use for slicing (axial, coronal or sagittal)
* --height: Slice height (256 for conformed volumes)
* --width: Slice width (256 for conformed volumes)
* --thickness: Number of pre- and suceeding slices (we use 3 --> total of 7 slices is fed to the network)

#### Example Command Axial
```
python3 generate_hdf5.py \
--hdf5_name ../data/training_set_cispa_axial.hdf5 \
--csv_file ../training_set_subjects_dirs.csv \
--plane axial \
--image_name mri/orig.mgz \
--gt_name mri/aparc.DKTatlas+aseg.mgz \
--gt_nocc mri/aseg.auto_noCCseg.mgz

```

#### Example Command Coronal
```
python3 generate_hdf5.py \
--hdf5_name ../data/training_set_cispa_coronal.hdf5 \
--csv_file ../training_set_subjects_dirs.csv \
--plane coronal \
--image_name mri/orig.mgz \
--gt_name mri/aparc.DKTatlas+aseg.mgz \
--gt_nocc mri/aseg.auto_noCCseg.mgz

```

#### Example Command Sagittal
```
python3 generate_hdf5.py \
--hdf5_name ../data/training_set_cispa_sagittal.hdf5 \
--csv_file ../training_set_subjects_dirs.csv \
--plane sagittal \
--image_name mri/orig.mgz \
--gt_name mri/aparc.DKTatlas+aseg.mgz \
--gt_nocc mri/aseg.auto_noCCseg.mgz

```

#### Example Command Sagittal using --data_dir instead of --csv_file
--data_dir specifies the path in which the data is located, with --pattern we can select subjects from the specified path. By default the pattern is "*" meaning all subjects will be selected.
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

Setting --pattern "*" will select all 19 subjects (subject1, ..., subject19).
Now, if only a subset should be used for the hdf5-file (e.g. subject 10 till subject19), this can be done by changing the --pattern flag to "subject1[0-9]": 

```
python3 generate_hdf5.py \
--hdf5_name ../data/training_set_cispa_axial.hdf5 \
--data_dir ../data \
--pattern "subject1[0-9]" \
--plane sagittal \
--image_name mri/orig.mgz \
--gt_name mri/aparc.DKTatlas+aseg.mgz \
--gt_nocc mri/aseg.auto_noCCseg.mgz
 
```

# 3. Training

The *FastSurferCNN* directory contains all the source code and modules needed to run the scripts. A list of python libraries used within the code can be found in __requirements.txt__. The main script is called __train.py__ within which certain options can be selected and set via the command line:

#### Training and Validation sets
* --hdf5_name_train: Path to training hdf5-dataset
* --hdf5_name_val: Path to validation hdf5-dataset
* --batch_size: Input batch size for training set (default: 16)
* --validation_batch_size: Input batch size for validation set (default: 16)

#### Training options
* --plane: which anatomical view should be trained on (axial (default), coronal or sagittal). Has to fit with the hdf5-sets.
* --epochs: Number of epochs to train (default=30)
* --lr: learning rate (default=0.01)
* --optim: Optimizer to use (adam (default) or sgd)
* --momentum: Momentum for optimizer (SGD)
* --nesterov: Enables Nesterov for optimizer (SGD)
* --scheduler: Type of learning rate scheduler to use (StepLR (default), None)

#### Model options
* --num_filters: Filter dimensions for Networks (all layers same, default=64)
* --num_classes: Number of classes to predict including background. Axial and coronal = 79 (default), Sagittal = 51
* --num_channels: Number of input channels. Default=7 for thick slices.
* --kernel_height: Height of Kernel (default 5)
* --kernel_width: Width of Kernel (default 5)
* --stride: Stride during convolution (default 1)
* --stride_pool: Stride during pooling (default 2)
* --pool: Size of pooling filter (default 2)

#### Logging
* --log-interval: How often should a model be saved (default every 2 epochs)
* --log_dir: directory to store checkpoints and validation confusion matrices in

#### Optional commands
* --resume: resume training from earlier time point
* --no-cuda: disable CUDA training
* --seed: Select random seed (default=1)


To train the network on a given hdf5-set, change into the *FastSurferCNN* directory and run one of the following commands: 

### Example Commands Network Training
```
# Sagittal view

python3 train.py \
--hdf5_name_train ../data/training_set_cispa_sagittal.hdf5 \
--hdf5_name_val ../data/validation_set_sagittal.hdf5 \
--plane sagittal \
--log_dir ../checkpoints/Sagittal_Competitive_APARC_ASEG/ \
--epochs 30 \
--num_channels 7 \
--num_classes 51
 
# Coronal view

python3 train.py \
--hdf5_name_train ../data/training_set_cispa_coronal.hdf5 \
--hdf5_name_val ../data/validation_set_coronal.hdf5 \
--plane coronal \
--log_dir ../checkpoints/Coronal_Competitive_APARC_ASEG/ \
--epochs 30 \
--num_channels 7 \
--num_classes 79

# Axial view

python3 train.py \
--hdf5_name_train ../data/training_set_cispa_axial.hdf5 \
--hdf5_name_val ../data/validation_set_axial.hdf5 \
--plane axial \
--log_dir ../checkpoints/Axial_Competitive_APARC_ASEG/ \
--epochs 30 \
--num_channels 7 \
--num_classes 79
```
