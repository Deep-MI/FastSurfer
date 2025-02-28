BATCH: brun_fastsurfer.sh
=========================

Usage
-----

```{command-output} ./brun_fastsurfer.sh --help
:cwd: /../
```

Subject Lists
-------------

The input files and options may be specified in three ways:

1. By writing them into the console (or by piping them in) (default) (one case per line),
2. by passing a subject list file `--subject_list <filename>` (one case per line), or
3. by passing them on the command line `--subjects "<subject_id>=<t1 file>" [more cases]` (no additional options 
   supported).

These files/input options will usually be in the format `<subject_id>=<t1 file> [additional options]`, where additional
options are optional and enable passing options different to the "general options" given on the command line to 
`brun_fastsurfer.sh`. One example for such a case-specific option is an optional T2w image (e.g. for the 
[HypVINN](../overview/OUTPUT_FILES.md#hypvinn-module)). An example subject list file might look like this:

```
001=/data/study/raw/T1w-001.nii.gz --t2 /data/study/raw/T2w-001.nii.gz
002=/data/study/raw/T1w-002.nii.gz --t2 /data/study/raw/T2w-002.nii.gz
002=/data/study/raw/T1w-003-alt.nii.gz --t2 /data/study/raw/T2w-003.nii.gz
... 
```

Parallelization with `brun_fastsurfer.sh`
-----------------------------------------

`brun_fastsurfer.sh` has powerful builtin parallel processing capabilities. These are hidden underneath the 
`--parallel* <n>|max` and the `--device <device>` as well as `--viewagg_device <device>` flags.
One of the core properties of FastSurfer is the split into the segmentation (which uses Deep Learning and therefore 
benefits from GPUs) and the surface pipeline (which does not benefit from GPUs). For ideal batch processing, we want 
different resource scheduling.

`--parallel*` allows three parallel batch processing modes: serial, single parallel pipeline and dual parallel pipeline. 

### Serial processing (default)
Each case/image is processed after the other fully, i.e. surface reconstruction of case 1 is fully finished before 
segmentation of case 2 is started. This setting is the default and represents the manual flags `--parallel 1`.

### Single parallel pipeline
This mode is ideal for CPU-based processing for segmentation. It will process segmentations and surfaces in series 
in the same process, but multiple cases are processed at the same time.

```bash
$FASTSURFER_HOME/brun_fastsurfer.sh --parallel 4 --threads 2
```
will start 4 segmentations (and surface reconstructions) at the same time, and will start a fifth, when the surface
processing of one of the four first cases is finished (`--parallel 4`). It will try to use 2 threads per case
(`--threads 2`) and perform reconstruction of left and right hemispheres in parallel (`--threads 2`, 2 >= 2).
`--parallel max` will remove the limit and start all cases at the same time (each with the target number of threads 
given by `--threads`).

### Dual parallel pipeline
This is ideal for GPU-based processing for segmentation. It will process segmentations and surfaces in separate 
pipelines, which is useful for optimized GPU loading. Multiple cases may be processed at the same time.

```bash
$FASTSURFER_HOME/brun_fastsurfer.sh --device cuda:0-1 --parallel_seg 2 --parallel_surf max \
  --threads_seg 8 --threads_surf 4
```
will start 2 parallel segmentations (`--parallel_seg 2`) using GPU 0 for case 1 and GPU 1 for case 2 
(`--device cuda:0-1` -- same as `--device cuda:0,1`). After one of these segmentations is finished, the segmentation of 
case 3 will start on that same device as well as the surface reconstruction (without putting a limit on parallel 
surface reconstructions, `--parallel_surf max`). Each segmentation process will aim to use 8 threads/cores
(`--threads_seg 8`) and each surface reconstruction process will aim to use 4 threads (`--threads_surf 4`) with both
hemispheres processed in parallel (`--threads_surf 4`, 4 >= 2, so right hemisphere will use 2 threads and left as well).

Note, if your GPU has sufficient [video memory](../overview/intro.rst#system-requirements), two parallel segmentations
can run on the same GPU, but the script cannot schedule more than one process per GPU for multiple GPUs, i.e.
`--device cuda:0,1 --parallel_seg 4` is not supported.

Questions
---------
Can I disable the progress bars in the output?

> You can disable the progress bars by setting the TQDM_DISABLE environment variable to 1, if you have tqdm>=4.66.
> 
> For docker, this can be done with the flag `-e`, e.g. `docker run -e TQDM_DISABLE=1 ...`, for singularity with the flag `--env`, e.g. `singularity exec --env TQDM_DISABLE=1 ...` and for native installations by prepending, e.g. `TQDM_DISABLE=1 ./run_fastsurfer.sh ...`.

