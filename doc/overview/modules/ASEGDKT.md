Whole-Brain Segmentation (FastSurferVINN)
==========================================
The core module of FastSurfer (`asegdkt`): a deep learning network (FastSurferVINN) that segments the whole brain into
95 cortical and subcortical structures, following FreeSurfer's DKT atlas, and computes volume statistics for them.
All other modules build on its result.

What it computes
----------------
- A segmentation of cortical and subcortical structures (`aparc.DKTatlas+aseg.deep.mgz`), equivalent to FreeSurfer's
  `aparc.DKTatlas+aseg.mgz`, and a simplified subcortical segmentation (`aseg.auto_noCCseg.mgz`).
- A brain mask and a bias-field corrected image.
- Volume statistics for all structures, corrected for partial volume effects (`aseg+DKT.stats`).
- Optionally, with `--tal_reg`, the estimated total intracranial volume (eTIV) in the statistics.

The output files are listed in the [output files overview](../OUTPUT_FILES.md#segmentation-module). If the
[surface reconstruction](SURFACE.md) also runs, it refines the segmentation along the cortex with the surfaces, and its
updated segmentations and statistics are the ones to use.

What it needs
-------------
- A T1-weighted image, see the [requirements to input images](../../../README.md#requirements-to-input-images).
- Voxel sizes between 0.7 mm and 1 mm are supported natively (smaller voxels are experimental), see `--vox_size`.
- No FreeSurfer license, unless you switch on the optional Talairach registration (`--tal_reg`), which is only used
  to estimate the total intracranial volume (eTIV); that needs a [FreeSurfer license](../INSTALL.md#freesurfer-license).

The segmentation takes a few minutes on a GPU, and longer on the CPU.

Options
-------
The module runs by default. The most important options of `run_fastsurfer.sh` for it:

- `--seg_only`: run the segmentation modules only, without the surface reconstruction.
- `--no_asegdkt`: skip this module, for example to rerun only other modules on an existing segmentation.
- `--asegdkt_segfile <path>`: where to write the segmentation (default `mri/aparc.DKTatlas+aseg.deep.mgz`).
- `--vox_size <0.7-1|min>`: the voxel size to process at (default `min`, from the input image).
- `--no_biasfield`: skip the bias-field correction and the partial-volume corrected statistics.
- `--tal_reg` (with `--seg_only`, optional): add the Talairach registration, only used to estimate eTIV; it needs a
  FreeSurfer license, and `--3T` uses the 3T atlas for it. The surface reconstruction always computes eTIV.
- `--keepgeom` (with `--seg_only`): write the outputs in the geometry of the input image.

Usage:

```text
run_fastsurfer.sh --seg_only --sd <subjects_dir> --sid <subject_id> --t1 <t1_path>
```

All options are described in the [run_fastsurfer.sh reference](../../scripts/RUN_FASTSURFER.md). To run the network
script directly, see [FastSurferVINN in the command reference](../../scripts/fastsurfercnn.rst).

References
----------
If you use FastSurfer in your research, please cite:

- Henschel L, Conjeti S, Estrada S, Diers K, Fischl B, Reuter M.
  **FastSurfer - A fast and accurate deep learning based neuroimaging pipeline.**
  *NeuroImage* 219 (2020), 117012.
  [doi:10.1016/j.neuroimage.2020.117012](https://doi.org/10.1016/j.neuroimage.2020.117012)
- Henschel L\*, Kuegler D\*, Reuter M. (\*co-first)
  **FastSurferVINN: Building Resolution-Independence into Deep Learning Segmentation Methods - A Solution for HighRes Brain MRI.**
  *NeuroImage* 251 (2022), 118933.
  [doi:10.1016/j.neuroimage.2022.118933](https://doi.org/10.1016/j.neuroimage.2022.118933)
