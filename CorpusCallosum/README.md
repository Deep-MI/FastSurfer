Corpus Callosum Pipeline
========================
FastSurfer-CC is a deep learning-based pipeline for automated segmentation, analysis, and shape analysis of the corpus callosum in brain MRI scans.
Also segments the fornix, localizes the anterior and posterior commissure (AC and PC) and standardizes the orientation of the brain.

The documentation is split into three files, please refer to:
- [Module Overview](../doc/overview/modules/CC.md): Description of the pipeline, and the corpus callosum measures produced.
- [Advanced options](../doc/scripts/fastsurfer_cc.rst): Quality control, custom subdivision and visualization options.
- [CC visualization](../doc/scripts/cc_visualization.rst): 2D/3D templates and standalone fsaverage value maps.
- [Output Files](../doc/overview/OUTPUT_FILES.md#corpus-callosum-module): List of output files and their descriptions.

Quickstart
----------
The expert command expects an existing FastSurfer subject containing at least `mri/orig.mgz` and
`mri/aseg.auto_noCCseg.mgz` by default. The input paths can be overridden with `--conformed_name` and `--aseg_name`.
It reads those inputs and writes all CC outputs into the subject directory.

### Native

```bash
python3 CorpusCallosum/fastsurfer_cc.py \
    --sd /data/subjects \
    --sid sub001 \
    --upright_volume mri/upright_volume.mgz \
    --qc_image qc_snapshots/callosum.png \
    --thickness_image qc_snapshots/callosum.thickness.png
```

Paths that are not absolute are resolved relative to `/data/subjects/sub001`. The upright volume is useful for
checking the midplane and is the anatomical reference for manual CC editing.

### Docker

The FastSurfer Docker image normally starts `run_fastsurfer.sh`. Expert commands therefore override the entrypoint
with FastSurfer's environment-setup wrapper and explicitly invoke `fastsurfer_cc.py`:

```bash
SUBJECTS_DIR=/data/fastsurfer
SID=sub001

docker run --gpus all --rm \
    --user "$(id -u):$(id -g)" \
    --volume "$SUBJECTS_DIR:/output" \
    --entrypoint /fastsurfer/tools/Docker/entrypoint.sh \
    deepmi/fastsurfer:latest \
    python3 /fastsurfer/CorpusCallosum/fastsurfer_cc.py \
    --sd /output \
    --sid "$SID" \
    --upright_volume mri/upright_volume.mgz \
    --qc_image qc_snapshots/callosum.png \
    --thickness_image qc_snapshots/callosum.thickness.png
```


### Singularity or Apptainer

Build or download an image as described in the [Singularity documentation](../doc/overview/SINGULARITY.md), then bind
the subject directory and invoke the expert script directly:

```bash
SUBJECTS_DIR=/data/fastsurfer
SID=sub001
FASTSURFER_SIF=/containers/fastsurfer-gpu.sif

singularity exec --nv --no-mount home,cwd -e \
    --bind "$SUBJECTS_DIR:/output" \
    "$FASTSURFER_SIF" \
    python3 /fastsurfer/CorpusCallosum/fastsurfer_cc.py \
    --sd /output \
    --sid "$SID" \
    --upright_volume mri/upright_volume.mgz \
    --qc_image qc_snapshots/callosum.png \
    --thickness_image qc_snapshots/callosum.thickness.png
```

The same command works with Apptainer by replacing `singularity` with `apptainer`. For CPU execution with the same
image, omit `--nv`; FastSurfer falls back to the CPU. `--no-mount home,cwd -e` prevents host Python packages and
environment variables from leaking into the container.

These commands generate all standard CC outputs. Morphometry is written to `stats/callosum.CC.midslice.json`,
including 100 thickness measurements and the areas of subsegments.

The expert interface also supports supplied 3D AC/PC voxel coordinates and manual corrections of the upright CC
segmentation. Copy `mri/callosum.CC.upright.mgz` to `mri/callosum.CC.upright.manedit.mgz`, edit label 192 using
`mri/upright_volume.mgz` as the reference, and pass the edited path to `--segmentation_manedit`. A manual segmentation
containing any fornix label-250 voxels is self-contained and authoritative for the entire fornix. If it has no label
250, the automatic upright and original-space CC segmentations from a previous run are required and supply the
fornix. Choose supplied AC/PC coordinates before creating the edit and reuse the exact same coordinates and midplane
method for every edit rerun; otherwise regenerate the upright reference and recreate or rebase the correction. See the
[advanced documentation](../doc/scripts/fastsurfer_cc.rst) for complete native, Docker, and Singularity/Apptainer
commands.
