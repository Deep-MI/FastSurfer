# Longitudinal Processing

FastSurfer has a dedicated pipeline to quantify longitudinal changes in T1-weighted MRI. FastSurfer's longitudinal pipeline outpeforms independent (cross sectional) processing of individual MRIs across time in both FastSurfer and FreeSurfer, as well as even the longitudional pipeline in FreeSurfer.

## What is Longitudinal Processing

In longitudinal studies, MRIs of the same participant are acquired at different time points. Usually the goal is to quantify potentially subtle anatomical changes representing early disease effects or effects of disease modifying therapies or drug studies. In these situations we know that most of the anatomy will be very similar, as compared to cross sectional differences between participants. Longitudinal processing, as opposed to independent processing of each MRI, tries to make use of the joint information to reduce variance across time, leading to more sensitive estimates of longitudinal changes. This methodological approach leads to increased statistical power to detect subtle changes and, therefore, permits to either find smaller effects or to reduce the number of particiants needed to detect such an effect - saving time and money. Our paper for the FreeSurfer longitudinal stream (Reuter et al. [2012](https://doi.org/10.1016/j.neuroimage.2012.02.084)) nicley highlights these advantages, such as increased reliability and sensitiviy and describes the general idea. 

Generally the idea is to: 
- Align images across time robustly into an unbiased mid-space (Reuter et al. [2010](https://doi.org/10.1016/j.neuroimage.2010.07.020)).
- Construct a template image for each participant (called within-subject template).
- Process the template image, e.g. to generate initial WM and GM surfaces.
- Process each time point, initializing or reusing results from the template, yet allowing enough freedom for results to evolve. 

This approach is used in FreeSurfer and in FastSurfer and it avoids multiple issues that are inherent to other approaches:
- It avoids the introduction of processing bias (Reuter, Fischl [2011](https://doi.org/10.1016/j.neuroimage.2011.02.076)) by treating all time points the same.
- It is independent on the number of time points, and independent of the time differences between acquisitions.
- It is flexible enough to not over-constrain (smooth) longitudinal effects.
- It does not enforce or encourage directional temporal changes (e.g. atrophy) and can therefore be used in studying cyclic patterns, or cross-over drug studies.

## How to Run Your Data

We are providing a new entry script `long_fastsurfer.sh` to help you process longitudinal data. 

```bash
# Setup FASTSURFER and FREESURFER
export FASTSURFER_HOME=/path/to/fastsurfer
export FREESURFER_HOME=/path/to/freesurfer
source $FREESURFER_HOME/SetUpFreeSurfer.sh

# Define data directory
export SUBJECTS_DIR=/home/user/my_fastsurfer_analysis

# Run FastSurfer longitudinally
$FASTSURFER_HOME/long_fastsurfer.sh \
    --tid <templateID> \
    --t1s <T1_1> <T1_2> ... \
    --tpids <tID1> <tID2> ...
```

Here `<templateID>` is a name you assign to this individual person and will be used in the output directory (`$SUBJECTS_DIR`) for the directory containing the within-subject template (e.g. "`--tid bert`"). The `<T1_1> <T1_2>` etc. are the global paths to the input full head T1w images for each time point (do not need to be bias corrected) in nifti or mgz format. The `<tID1> <tID2>` etc. are the ID names for each time point. Corresponding directories will be created in the output directory  (`$SUBJECTS_DIR`) , e.g. "`--tpids bert_1 bert_2`". These directories will contain the final results for each time point for downstream analysis.

Note, with a few exceptions, you can add additional flags that can be understood by `run_fastsurfer.sh`, which will be passed through, e.g. the `--3T` when working with 3T images. 

The above command will, of course, be slightly different when using the preferred installation way Singularity (or Docker). For example for Singularity:

```bash
singularity exec --nv \
                 --no-home \
                 -B /home/user/my_mri_data:/data \
                 -B /home/user/my_fastsurfer_analysis:/output \
                 -B /home/user/my_fs_license_dir:/fs_license \
                 ./fastsurfer-gpu.sif \
                 /fastsurfer/long_fastsurfer.sh \
                 --fs_license /fs_license/license.txt \
                 --tid <templateID> \
                 --t1s <T1_1> <T1_2> ... \
                 --tpids <tID1> <tID2> ... \
                 --sd /output \
                 --3T
```

## Single Time Point Cases

Sometimes your longitudinal data set contains participants with only one time point, e.g. due to drop-out or QC exclusion. Instead of excluding single-time point cases completely (which may even bias results), you can include them for better statistics. While this obviously will not help to better estimate longitudinal slopes, linear mixed effects models (LMEs), for example, can include single time point data to obtain better estimates of cross-subject variance. 

HOWEVER, this requires that you process these cases also through the longitudinal stream! This is very important, to ensure that they undergo the same processing steps as data from cases with multiple time points. Only then are the results comparable. The command is the same as above, just specify only the single t1 and time point id. Could not be any easier. 

## Behind the Scenes

`long_fastsurfer.sh` is just a helper script and will perform the following individual steps for you:
1. [Template Init] It will prepare the subject template by calling `long_prepare_template.sh`:
   ```bash
   long_prepare_template.sh \
     --tid <templateID> \
     --i1s <T1_1> <T1_2> ... \
     --tpids <tID1> <tID2>
   ```
   This will register (align) all time point images into the unbiased mid-space using `mri_robust_template`, after an initial segmentation and skull stripping. It will also create the template image, kind of a mean image across time. For single time point cases it will align the input into a standard upright position.
2. [Template Seg] Next, the template image will be segmented via a call to `run_fastsurfer.sh --sid <templateID> --base --seg_only ...` where the `--base` flag indicates that the input image will be taken from the already existing template directory. 
3. [Template Surf] This is followed by the surface processing of the template  `run_fastsurfer.sh --sid <templateID> --base --surf_only ...`, which can be combined with the previous step.
4. [Long Seg] Next, the segmentation of each time point, which can theoretically run in parallel with the previous two steps, is performed `run_fastsurfer.sh --sid <tIDn> --long <templateID> --seg_only ...`,
5. [Long Surf] Again followed by the surface processing for each time point: `run_fastsurfer.sh --<id <tIDn> --long <templateID> --surf_only`. This step needs to wait until 3. and 4. (for this time point) are finished. In this step, for example, surfaces are initialized with the ones obtained on the template above and only fine-tuned, instead of recreating them. 

Internally we use `brun_fastsurfer.sh` as a helper script to process multiple time points in parallel (in the LONG steps 4. and 5.). Here `--parallel_seg` can be passed to `long_fastsurfer.sh` to specify the number of parallel runs during the segmentation step (4) which is usually limited by GPU memory, if run on the GPU. Further `--parallel_surf` specifies the number of parallel surface runs on the CPU and is most impactful. It can be combined with `--threads_surf 2` (or higher) to switch on parallelization of the two hemispheres in each surface block. 

## Final Statistics:

The final results will be located in `$SUBJECTS_DIR/tID1` ... for each time point. These directories will have the same structure as a regular FastSurfer/FreeSurfer output directory. Therefore, you can use the regular downstream analysis tools, e.g. to extract statistics from the stats files. Note, that the surfaces are already in vertex-correspondence across time for each participant. For group analysis one would still need to map thickness estimates to fsaverage spherical template (this is usually done with `mris_preproc`). For longitudinal statistics using the (recommended) linear mixed effects models see our R toolbox [FS LME R](https://github.com/Deep-MI/fslmer), which can also analyze the mass-univariate situation for e.g. cortical thickness maps. Alternatively use this Matlab package: [LME Matlab](https://github.com/NeuroStats/lme) and our matlab tools for time-to-even (survival) analysis: [Survival](https://github.com/NeuroStats/Survival).

## References

- Reuter, Schmansky, Rosas, Fischl
  Within-subject template estimation for unbiased longitudinal image analysis.
  NeuroImage 61(4):1402-1418
  [https://doi.org/10.1016/j.neuroimage.2012.02.084](https://doi.org/10.1016/j.neuroimage.2012.02.084)
- Reuter, Fischl (2011).
  Avoiding asymmetry-induced bias in longitudinal image processing.
  NeuroImage 57(1):19-21
  [https://doi.org/10.1016/j.neuroimage.2011.02.076](https://doi.org/10.1016/j.neuroimage.2011.02.076)
- Reuter, Rosas, Fischl (2010). 
  Highly accurate inverse consistent registration: a robust approach. 
  NeuroImage 53(4):1181-1196
  [https://doi.org/10.1016/j.neuroimage.2012.02.084](https://doi.org/10.1016/j.neuroimage.2012.02.084)
- Diers, Reuter
  FreeSurfer and FastSurfer Linear Mixed Effects tools for R.
  [https://github.com/Deep-MI/fslmer](https://github.com/Deep-MI/fslmer)
- Sabuncu, Bernal-Rusiel, Greve, Reuter, Fischl (2014).
  Event time analysis of longitudinal neuroimage data.
  Neuroimage 97, 9-18
  [https://doi.org/10.1016/j.neuroimage.2014.04.015](https://doi.org/10.1016/j.neuroimage.2014.04.015)
- Bernal-Rusiel, Greve, Reuter, Fischl, Sabuncu (2013).
  Spatiotemporal Linear Mixed Effects Modeling for the Mass-univariate Analysis of Longitudinal Neuroimage Data.
  NeuroImage 81, 358-370
  [https://doi.org/10.1016/j.neuroimage.2013.05.049](https://doi.org/10.1016/j.neuroimage.2013.05.049)
- Bernal-Rusiel, Greve, Reuter, Fischl, Sabuncu (2012).
  Statistical Analysis of Longitudinal Neuroimage Data with Linear Mixed Effects Models.
  Neuroimage 66, 249-260
  [https://doi.org/10.1016/j.neuroimage.2012.10.065](https://doi.org/10.1016/j.neuroimage.2012.10.065)
