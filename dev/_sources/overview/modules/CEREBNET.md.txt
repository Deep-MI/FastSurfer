Cerebellum Sub-Segmentation (CerebNet)
======================================
The cerebellum module (`cereb`): a deep learning network (CerebNet) that divides the cerebellum into its lobules and
separates gray and white matter, and computes volume statistics for them.

What it computes
----------------
- A sub-segmentation of the cerebellum with a detailed delineation of gray and white matter
  (`cerebellum.CerebNet.nii.gz`).
- Volume statistics for the cerebellar structures, corrected for partial volume effects
  (`cerebellum.CerebNet.stats`).

The output files are listed in the [output files overview](../OUTPUT_FILES.md#cerebnet-module).

What it needs
-------------
- A T1-weighted image, see the [requirements to input images](../../../README.md#requirements-to-input-images).
- The whole-brain segmentation of the [FastSurferVINN module](ASEGDKT.md), which locates the cerebellum.
- CerebNet works at 1 mm: images with smaller voxels are resampled to 1 mm for it, and its outputs are at 1 mm.

The module adds a few minutes on a GPU to the segmentation.

Options
-------
The module runs by default, as part of the segmentation. The options of `run_fastsurfer.sh` for it:

- `--no_cereb`: skip this module.
- `--cereb_segfile <path>`: where to write the segmentation (default `mri/cerebellum.CerebNet.nii.gz`).
- `--no_biasfield`: skip the partial-volume corrected statistics.

All options are described in the [run_fastsurfer.sh reference](../../scripts/RUN_FASTSURFER.md). To run the network
script directly, see [CerebNet in the command reference](../../scripts/cerebnet.rst).

References
----------
If you use the cerebellum sub-segmentation in your research, please cite:

- Faber J\*, Kuegler D\*, Bahrami E\*, et al. (\*co-first)
  **CerebNet: A fast and reliable deep-learning pipeline for detailed cerebellum sub-segmentation.**
  *NeuroImage* 264 (2022), 119703.
  [doi:10.1016/j.neuroimage.2022.119703](https://doi.org/10.1016/j.neuroimage.2022.119703)
