Surface Reconstruction (recon-surf)
===================================
The surface pipeline (`recon-surf`): it reconstructs the cortical surfaces from the whole-brain segmentation, maps the
cortical parcellation onto them, and computes cortical thickness and other measures, as FreeSurfer's `recon-all` does,
but much faster.

What it computes
----------------
- The white matter and pial surfaces of both hemispheres, and inflated and spherical surfaces.
- Cortical thickness, area, volume and curvature for every surface point.
- The cortical parcellation mapped onto the surfaces, with statistics per region.
- A registration to FreeSurfer's `fsaverage` (`sphere.reg`), for comparing thickness maps across subjects.
- Segmentations and statistics refined with the surfaces, which replace those of the segmentation modules.
- All files that FreeSurfer's downstream tools expect, under FreeSurfer's names (see
  [FreeSurfer downstream modules](../intro.rst#freesurfer-downstream-modules)).

The output files are listed in the [output files overview](../OUTPUT_FILES.md#surface-module).

What it needs
-------------
- The outputs of the [whole-brain segmentation](ASEGDKT.md) and of the [corpus callosum module](CC.md), which the same
  run can compute.
- A [FreeSurfer license](../INSTALL.md#freesurfer-license), because it uses FreeSurfer tools.
- Voxel sizes between 0.7 mm and 1 mm are supported (smaller voxels are experimental); below 1 mm, the pipeline runs in
  its high-resolution mode.

The surface reconstruction takes about 20 to 40 minutes, because it processes both hemispheres in parallel by default.
With `--threads 1`, it processes them one after the other and takes considerably longer.

Options
-------
The surface reconstruction runs by default, after the segmentation. The most important options of
`run_fastsurfer.sh` for it:

- `--seg_only`: skip the surface reconstruction.
- `--surf_only`: run only the surface reconstruction, on an existing segmentation.
- `--3T`: use the 3T atlas for the Talairach registration, for better eTIV estimates of 3T images.
- `--threads_surf <n>`: the number of threads for this part.
- `--no_surfreg`: skip the registration to `fsaverage` (saves time, but is needed for cross-subject analyses of
  thickness maps; [recon-surfreg.sh](../../scripts/recon_surfreg.rst) adds it later).
- `--fsaparc`: also compute FreeSurfer's own cortical parcellation, in addition to the mapped one.

Usage:

```text
run_fastsurfer.sh --surf_only --sd <subjects_dir> --sid <subject_id> \
    --fs_license <freesurfer_license_path>
```

All options are described in the [run_fastsurfer.sh reference](../../scripts/RUN_FASTSURFER.md). To run
`recon-surf.sh` directly, see [recon-surf in the command reference](../../scripts/recon_surf.rst).

References
----------
If you use the surface reconstruction in your research, please cite:

- Henschel L, Conjeti S, Estrada S, Diers K, Fischl B, Reuter M.
  **FastSurfer - A fast and accurate deep learning based neuroimaging pipeline.**
  *NeuroImage* 219 (2020), 117012.
  [doi:10.1016/j.neuroimage.2020.117012](https://doi.org/10.1016/j.neuroimage.2020.117012)

The surface reconstruction is largely based on FreeSurfer, see
[how to cite FreeSurfer](https://surfer.nmr.mgh.harvard.edu/fswiki/FreeSurferMethodsCitation).
