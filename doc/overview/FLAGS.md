# FastSurfer_Flags
Next, you will need to select the `*fastsurfer-flags*` and replace `*fastsurfer-flags*` with your options. Please see the Examples below for some example flags.

The `*fastsurfer-flags*` will usually include the subject directory (`--sd`; Note, this will be the mounted path - `/output` - for containers), the subject name/id (`--sid`) and the path to the input image (`--t1`). For example:

```bash
... --sd /output --sid test_subject --t1 /data/test_subject_t1.nii.gz --3T
```
Additionally, you can use `--seg_only` or `--surf_only` to only run a part of the pipeline or `--no_biasfield`, `--no_cereb` and `--no_asegdkt` to switch off some segmentation modules (see above).
Here, we have also added the `--3T` flag, which tells fastsurfer to register against the 3T atlas for ICV estimation (eTIV).

In the following, we give an overview of the most important options, but you can view a full list of options with 

```bash
./run_fastsurfer.sh --help
```


## Required arguments
* `--sd`: Output directory \$SUBJECTS_DIR (equivalent to FreeSurfer setup --> $SUBJECTS_DIR/sid/mri; $SUBJECTS_DIR/sid/surf ... will be created).
* `--sid`: Subject ID for directory inside \$SUBJECTS_DIR to be created ($SUBJECTS_DIR/sid/...)
* `--t1`: T1 full head input (not bias corrected, global path). The network was trained with conformed images (UCHAR, 256x256x256, 1-0.7 mm voxels and standard slice orientation). These specifications are checked in the run_prediction.py script and the image is automatically conformed if it does not comply. Note, outputs will be in the conformed space (as in FreeSurfer). 

## Required for docker when running surface module
* `--fs_license`: Path to FreeSurfer license key file (only needed for the surface module). Register (for free) at https://surfer.nmr.mgh.harvard.edu/registration.html to obtain it if you do not have FreeSurfer installed so far. Strictly necessary if you use Docker, optional for local install (your local FreeSurfer license will automatically be used). The license file is usually located in $FREESURFER_HOME/license.txt or $FREESURFER_HOME/.license . 

## Segmentation pipeline arguments (optional)
* `--seg_only`: only run FastSurferCNN (generate segmentation, do not run the surface pipeline)
* `--seg_log`: Name and location for the log-file for the segmentation (FastSurferCNN). Default: $SUBJECTS_DIR/$sid/scripts/deep-seg.log
* `--viewagg_device`: Define where the view aggregation should be run on. Can be "auto" or a device (see --device). By default, the program checks if you have enough memory to run the view aggregation on the gpu. The total memory is considered for this decision. If this fails, or you actively overwrote the check with setting with "cpu" view agg is run on the cpu. Equivalently, if you pass a different device, view agg will be run on that device (no memory check will be done).
* `--device`: Select device for NN segmentation (_auto_, _cpu_, _cuda_, _cuda:<device_num>_, _mps_), where cuda means Nvidia GPU, you can select which one e.g. "cuda:1". Default: "auto", check GPU and then CPU. "mps" is for native MAC installs to use the Apple silicon (M-chip) GPU. 
* `--asegdkt_segfile`: Name of the segmentation file, which includes the aparc+DKTatlas-aseg segmentations. Requires an ABSOLUTE Path! Default location: \$SUBJECTS_DIR/\$sid/mri/aparc.DKTatlas+aseg.deep.mgz
* `--no_cereb`: Switch of the cerebellum sub-segmentation
* `--cereb_segfile`: Name of the cerebellum segmentation file. If not provided, this intermediate DL-based segmentation will not be stored, but only the merged segmentation will be stored (see --main_segfile <filename>). Requires an ABSOLUTE Path! Default location: \$SUBJECTS_DIR/\$sid/mri/cerebellum.CerebNet.nii.gz
* `--no_biasfield`: Deactivate the calculation of partial volume-corrected statistics.

## Surface pipeline arguments (optional)
* `--surf_only`: only run the surface pipeline recon_surf. The segmentation created by FastSurferCNN must already exist in this case.
* `--3T`: for Talairach registration, use the 3T atlas instead of the 1.5T atlas (which is used if the flag is not provided). This gives better (more consistent with FreeSurfer) ICV estimates (eTIV) for 3T and better Talairach registration matrices, but has little impact on standard volume or surface stats.
* `--fstess`: Use mri_tesselate instead of marching cube (default) for surface creation
* `--fsqsphere`: Use FreeSurfer default instead of novel spectral spherical projection for qsphere
* `--fsaparc`: Use FS aparc segmentations in addition to DL prediction (slower in this case and usually the mapped ones from the DL prediction are fine)
* `--parallel`: Run both hemispheres in parallel
* `--no_fs_T1`: Do not generate T1.mgz (normalized nu.mgz included in standard FreeSurfer output) and create brainmask.mgz directly from norm.mgz instead. Saves 1:30 min.
* `--no_surfreg`: Skip the surface registration (`sphere.reg`), which is generated automatically by default. To safe time, use this flag to turn this off.

## Other
* `--threads`: Target number of threads for all modules (segmentation, surface pipeline), `1` (default) forces FastSurfer to only really use one core. Note, that the default value may change in the future for better performance on multi-core architectures.
* `--vox_size`: Forces processing at a specific voxel size. If a number between 0.7 and 1 is specified (below is experimental) the T1w image is conformed to that isotropic voxel size and processed. 
  If "min" is specified (default), the voxel size is read from the size of the minimal voxel size (smallest per-direction voxel size) in the T1w image:
  If the minimal voxel size is bigger than 0.98mm, the image is conformed to 1mm isometric.
  If the minimal voxel size is smaller or equal to 0.98mm, the T1w image will be conformed to isometric voxels of that voxel size.
  The voxel size (whether set manually or derived) determines whether the surfaces are processed with highres options (below 1mm) or not.
* `--py`: Command for python, used in both pipelines. Default: python3.10
* `--conformed_name`: Name of the file in which the conformed input image will be saved. Default location: \$SUBJECTS_DIR/\$sid/mri/orig.mgz
* `--ignore_fs_version`: Switch on to avoid check for FreeSurfer version. Program will terminate if the supported version (see recon-surf.sh) is not sourced. Can be used for testing dev versions.
* `-h`, `--help`: Prints help text