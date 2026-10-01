run_fastsurfer.sh
=================
Next, you will learn how to specify the `*fastsurfer-flags*` by replacing `*fastsurfer-flags*` with your specific options.
`run_fastsurfer.sh` is the central command of FastSurfer. In general, `run_fastsurfer.sh` is called once for each T1w MRI image that is to be processed and each call will result in one "Subject Folder" with segmentation maps, surfaces and statistics tables. If you want to process multiple images, you can either loop through the images yourself or use [brun_fastsurfer.sh](BATCH.md) or [srun_fastsurfer.sh](SLURM.md), which are multi-subject extensions to `run_fastsurfer.sh`.

On this page, we explain FastSurfer's options, usually referred to as `<*fastsurfer-flags*>` in this documentation.
The `<*fastsurfer-flags*>` will usually at least include the subject directory (`--sd`), the subject name/id (`--sid`) and the path to the input image (`--t1`). For example:

```bash
$FASTSURFER_HOME/run_fastsurfer.sh --sd $HOME/my_fastsurfer_data --sid test_subject --t1 $HOME/my_mri_data/test_subject_t1.nii.gz --3T
```
Additionally, you can use `--seg_only` or `--surf_only` to only run a part of the pipeline or `--no_biasfield`, `--no_cereb`, `--no_hypothal`, `--no_cc`, and `--no_asegdkt` to switch off individual segmentation modules.
Here, we have also added the `--3T` flag, which tells FastSurfer to register against the 3T atlas which is only relevant for the ICV estimation (eTIV).

In the following, we give an overview of the most important options. You can view a [full list of options](RUN_FASTSURFER.md#full-list-of-flags) with

```bash
./run_fastsurfer.sh --help
```

Required arguments
------------------
* `--sd`: Output directory \$SUBJECTS_DIR (equivalent to FreeSurfer setup --> $SUBJECTS_DIR/sid/mri; $SUBJECTS_DIR/sid/surf ... will be created).
* `--sid`: Subject ID for directory inside \$SUBJECTS_DIR to be created ($SUBJECTS_DIR/sid/...)
* `--t1`: T1 full head input (does not need to be bias corrected, global path). The network was trained with conformed images (UCHAR, cubic volume, 0.7mm - 1mm voxels and standard slice orientation; typically 256x256x256 at 1mm and larger cubes for higher-resolution isotropic inputs). These specifications are checked in the run_prediction.py script and the image is automatically conformed if it does not comply. By default, outputs are written in the FastSurfer conform space used for segmentation, which closely follows FreeSurfer conforming in `mri_convert -c`. The `--keepgeom` path is the exception: it uses an internal soft-LIA reordering for the 2D networks and maps results back to native geometry before writing outputs.

### Conditionally required
Required for Docker when running surface module:
* `--fs_license`: Path to FreeSurfer license key file (needed for the surface module and, if activated, the talairach registration `--tal_reg` in the segmentation). For local installs, your local FreeSurfer license will automatically be detected (usually `$FREESURFER_HOME/license.txt` or `$FREESURFER_HOME/.license`). Use this flag if autodetection fails or if you use Docker with the surface module. To get a license, [register (for free)](https://surfer.nmr.mgh.harvard.edu/registration.html).

Optional arguments
------------------------------------------
### Segmentation pipeline arguments
* `--seg_only`: Only run the brain segmentation pipeline and skip the surface pipeline.
* `--seg_log`: Name and location for the log-file for the segmentation. Default: $SUBJECTS_DIR/$sid/scripts/deep-seg.log
* `--viewagg_device`: Define where the view aggregation should be run on. Can be "auto" or a device (see --device). By default, the program checks if you have enough memory to run the view aggregation on the GPU. The total memory is considered for this decision. If this fails, or you actively specify "cpu" view aggregation is run on the CPU. Equivalently, if you pass a different device, view aggregation will be run on that device (no memory check will be done).
* `--device`: Select device for neural network segmentation (_auto_, _cpu_, _cuda_, _cuda:<device_num>_, _mps_), where cuda means Nvidia GPU, you can select which one e.g. "cuda:1". Default: "auto", check GPU and then CPU. "mps" is for native MAC installs to use the Apple silicon (M-chip) GPU.
* `--asegdkt_segfile`: Name of the segmentation file, which includes the aparc+DKTatlas-aseg segmentations. Requires an ABSOLUTE Path! Default location: \$SUBJECTS_DIR/\$sid/mri/aparc.DKTatlas+aseg.deep.mgz
* `--no_cereb`: Switch off the cerebellum sub-segmentation.
* `--no_hypothal`: Skip the hypothalamus segmentation.
* `--no_cc`: Skip the segmentation and analysis of the corpus callosum.
* `--lesion_mask <path to file>`: Path to a binary lesion mask in the same space as the T1 input. If provided, FastSurfer will wrap the segmentation and surface pipelines with lesion inpainting using LIT. This experimental feature is useful for images with tumors or other large lesions; review LIT-modified outputs before downstream use.
* `--cereb_segfile`: Name of the cerebellum segmentation file. Requires an ABSOLUTE Path! Default location: \$SUBJECTS_DIR/\$sid/mri/cerebellum.CerebNet.nii.gz
* `--no_biasfield`: Deactivate the biasfield correction and calculation of partial volume-corrected statistics in the segmentation modules. HypVINN does run but expects that biasfields are corrected externally.
* `--native_image` or `--keepgeom`: **Only supported for `--seg_only`**. Preserve the native image geometry (orientation, image size, and voxel size) for saved outputs. Internally, FastSurfer may temporarily reorder/flip the image to a soft-LIA layout so the 2D networks still see the expected plane ordering, but written outputs stay in native geometry; only intensity scaling and dtype conversion are applied as needed. This also includes experimental support for anisotropic images (no extreme anisotropy).

### Surface pipeline arguments
* `--surf_only`: Only run the surface pipeline. The segmentation created by FastSurferVINN must already exist in this case.
* `--3T`: Only affects Talairach registration: use the 3T atlas instead of the 1.5T atlas (which is used if the flag is not provided). This gives better (more consistent with FreeSurfer) ICV estimates (eTIV) for 3T and better Talairach registration matrices, but has little impact on standard volume or surface stats.
* `--fstess`: Use mri_tesselate instead of marching cube (default) for surface creation (not recommended, but more similar to FreeSurfer)
* `--fsqsphere`: Use FreeSurfer default instead of novel spectral spherical projection for qsphere (also not recommended)
* `--fsaparc`: Use FS aparc segmentations in addition to DL prediction (slower in this case and usually the mapped ones from the DL prediction are fine)
* `--no_fs_T1`: Skip generation of `T1.mgz` (normalized `nu.mgz` included in standard FreeSurfer output) and create `brainmask.mgz` directly from `norm.mgz` instead. Saves 1:30 min.
* `--no_surfreg`: Skip the surface registration (which creates `sphere.reg`) to safe time. Note, `sphere.reg` will be needed for any cross-subject statistical analysis of thickness maps, so do not use this option if you plan to perform cross-subject analysis.

### Some other flags
* `--threads`, `--threads_seg` and `--threads_surf`: Target number of threads for all modules, segmentation, and surface pipeline. Defaults: 1 for segmentation, 2 for surfaces.
  For surfaces the value is a *total* budget: with 2 or more the two hemispheres run at the same time and split it, so the default of 2 gives one thread each and 8 gives four each. `--parallel` runs the hemispheres at the same time with one thread each even at `--threads 1`, which keeps every binary single threaded, and so reproducible, while still using two cores; above 1 it has no effect.
  The topology correction always runs single-threaded regardless, because its result depends on the processing order (see [Reproducibility](#reproducibility)).
* `--vox_size`: Forces processing at a specific voxel size. If a number between 0.7 and 1 is specified (below is experimental) the T1w image is conformed to that isotropic voxel size and processed.
  If "min" is specified (default), the voxel size is read from the size of the minimal voxel size (smallest per-direction voxel size) in the T1w image:
  If the minimal voxel size is bigger than 0.98mm, the image is conformed to 1mm isotropic.
  If the minimal voxel size is smaller or equal to 0.98mm, the T1w image will be conformed to isotropic voxels of that voxel size.
  The voxel size (whether set manually or derived) determines whether the surfaces are processed with highres options (below 1mm) or not.
* `--py`: Command for python, used in both pipelines. Default: python3
* `--conformed_name`: Name of the file in which the conformed input image will be saved. Default location: \$SUBJECTS_DIR/\$sid/mri/orig.mgz
* `-h`, `--help`: Prints help text

Reproducibility
---------------

### On one machine

Re-running the same input on the same machine, with the same flags, the same thread count and the
same FastSurfer and FreeSurfer versions, is expected to give the same result: two runs at the
default of two threads came out identical in every file we compare.

Within one machine the only source of deviation we know of is the topology correction, and that now
always runs single-threaded. Other steps have not been tested at every thread count, so if you need
certainty, use `--threads 1`, or `--threads 1 --parallel` to keep every binary single threaded while
still processing the two hemispheres at the same time.

### Across machines

Numerical libraries choose their kernels from what the processor offers, so the same code can take
a different path on a different machine. Two layers do this independently, and both had to be
addressed:

- **torch**, in the segmentation networks. The vector instruction set decides the convolution
  kernel, and within one instruction set the vendor can still decide the code branch. Capping the
  instruction set alone leaves a difference between vendors; fixing the branch alone leaves a
  difference between instruction sets. Both are needed.
- **LAPACK and BLAS**, through numpy, in the surface pipeline and the registrations. These pick
  kernels the same way, independently of torch. A difference here is small, but the topology
  correction turns a rounding difference into a different retessellation that every later surface
  inherits, which is how a change far below single precision becomes a visible one.

Only the second of these is pinned for you. The spherical projection, the talairach registration
and the corpus callosum step set the numpy and OpenBLAS variables themselves, because that is where
a rounding difference was found to become a structural one. Nothing sets the torch variables: they
would slow every segmentation down for everyone, and most runs do not need to match another
machine.

**So a default run is not reproducible across machines.** If you need that, set all of them
yourself for the whole run:

```bash
# the numpy and OpenBLAS values depend on the host, so read them from the helper
eval "$(python $FASTSURFER_HOME/recon_surf/pin_cpu_dispatch.py)"
export ATEN_CPU_CAPABILITY=avx2 ONEDNN_MAX_CPU_ISA=AVX2 MKL_CBWR=COMPATIBLE
```

Pass them into the container with `--env` if you run FastSurfer with docker or singularity. This is
the configuration we test, and with it two x86-64 machines of different vendors produce identical
output. The cost is some speed, because the faster kernels are the ones being declined.

**It also covers the places we found, not every place that could exist.** Any numpy or BLAS call in
a step we have not examined is still free to dispatch on the hardware, which is the other reason to
set these globally rather than to rely on the three steps that pin themselves.

**GPU is not covered.** None of this applies to `--device cuda`: the card model and the precision
modes it selects are a separate source, and we have not tested it. A CPU run and a GPU run of the
same input are not expected to agree, so `--device` is part of what you have to hold constant.

Use one container image as well, see [Singularity](../overview/SINGULARITY.md). On macOS the
FreeSurfer binaries are built without OpenMP and run single-threaded regardless, and that
combination is untested for cross-machine agreement.

To see which kernels a run actually chose, look at the host block near the top of
`scripts/deep-seg.log`. It records the CPU, the instruction set torch selected, the dispatch
overrides in force and a numerical fingerprint. The fingerprint is the reliable key for "was this
comparable hardware": the CPU model name is not, because the same model can expose different
features on different hosts.

To compare two runs, use `tools/compare_subjects.py`: it compares voxels, vertices, transforms,
statistics and labels rather than raw bytes, and reports how large each difference is. `diff` and
checksums are no use here, because the headers record timestamps and the command line, so identical
runs still differ byte for byte.

```bash
python tools/compare_subjects.py $SUBJECTS_DIR/subject_a $SUBJECTS_DIR/subject_b
```

Full list of flags
------------------
```{command-output} ./run_fastsurfer.sh --help
:cwd: /../
```
