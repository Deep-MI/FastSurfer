# Output files

## Segmentation module

The segmentation module outputs the files shown in the table below. The two primary output files are the `aparc.DKTatlas+aseg.deep.mgz` file, which contains the FastSurfer segmentation of cortical and subcortical structures based on the DKT atlas, and the `aseg+DKT.stats` file, which contains summary statistics for these structures. Note, that the surface model (downstream) corrects these segmentations along the cortex with the created surfaces. So if the surface model is used, it is recommended to use the updated segmentations and stats (see below). 

| directory | filename                     | module  | description                                                        |
|:----------|------------------------------|---------|--------------------------------------------------------------------|
| mri       | aparc.DKTatlas+aseg.deep.mgz | asegdkt | cortical and subcortical segmentation                              |
| mri       | aseg.auto_noCCseg.mgz        | asegdkt | simplified subcortical segmentation without corpus callosum labels |
| mri       | mask.mgz                     | asegdkt | brainmask                                                          |
| mri       | orig.mgz                     | asegdkt | conformed image                                                    |
| mri       | orig_nu.mgz                  | asegdkt | biasfield-corrected image                                          |
| mri/orig  | 001.mgz                      | asegdkt | original image                                                     |
| scripts   | deep-seg.log                 | asegdkt | logfile                                                            |
| stats     | aseg+DKT.stats               | asegdkt | table of cortical and subcortical segmentation statistics          |

## Cerebnet module

The cerebellum module outputs the files in the table shown below. Unless switched off by the `--no_cereb` argument, this module is automatically run whenever the segmentation module is run. It adds two files, an image with the sub-segmentation of the cerebellum and a text file with summary statistics.


| directory | filename                   | module   | description                                 |
|:----------|----------------------------|----------|---------------------------------------------|
| mri       | cerebellum.CerebNet.nii.gz | cerebnet | cerebellum sub-segmentation                 |
| stats     | cerebellum.CerebNet.stats  | cerebnet | table of cerebellum segmentation statistics |


## Surface module

The surface module is run unless switched off by the `--seg_only` argument. It outputs a large number of files, which generally correspond to the FreeSurfer nomenclature and definition. A selection of important output files is shown in the table below, for the other files, we refer to the [FreeSurfer documentation](https://surfer.nmr.mgh.harvard.edu/fswiki). In general, the "mri" directory contains images, including segmentations, the "surf" folder contains surface files (geometries and vertex-wise overlay data), the "label" folder contains cortical parcellation labels, and the "stats" folder contains tabular summary statistics. Many files are available for the left ("lh") and right ("rh") hemisphere of the brain. Symbolic links are created to map FastSurfer files to their FreeSurfer equivalents, which may need to be present for further processing (e.g., with FreeSurfer downstream modules). 

After running this module, some of the initial segmentations and corresponding volume estimates are fine-tuned (e.g., surface-based partial volume correction, addition of corpus callosum labels). Specifically, this concerns the `aseg.mgz `, `aparc.DKTatlas+aseg.mapped.mgz`, `aparc.DKTatlas+aseg.deep.withCC.mgz`, which were originally created by the segmentation module or have earlier versions resulting from that module.

The primary output files are pial, white, and inflated surface files, the thickness overlay files, and the cortical parcellation (annotation) files. The preferred way of assessing this output is the [FreeView](https://surfer.nmr.mgh.harvard.edu/fswiki/FreeviewGuide) software. Summary statistics for volume and thickness estimates per anatomical structure are reported in the stats files, in particular the `aseg.stats`, and the left and right `aparc.DKTatlas.mapped.stats` files. 

| directory | filename                                                       | module  | description                                                                                  |
|:----------|----------------------------------------------------------------|---------|----------------------------------------------------------------------------------------------|
| mri       | aparc.DKTatlas+aseg.deep.withCC.mgz                            | surface | cortical and subcortical segmentation incl. corpus callosum after running the surface module |
| mri       | aparc.DKTatlas+aseg.mapped.mgz                                 | surface | cortical and subcortical segmentation after running the surface module                       |
| mri       | aparc.DKTatlas+aseg.mgz                                        | surface | symlink to aparc.DKTatlas+aseg.mapped.mgz                                                    |
| mri       | aparc+aseg.mgz                                                 | surface | symlink to aparc.DKTatlas+aseg.mapped.mgz                                                    |
| mri       | aseg.mgz                                                       | surface | subcortical segmentation after running the surface module                                    |
| mri       | wmparc.DKTatlas.mapped.mgz                                     | surface | white matter parcellation                                                                    |
| mri       | wmparc.mgz                                                     | surface | symlink to wmparc.DKTatlas.mapped.mgz                                                        |
| surf      | lh.area, rh.area                                               | surface | surface area overlay file                                                                    |
| surf      | lh.curv, rh.curv                                               | surface | curvature overlay file                                                                       |
| surf      | lh.inflated, rh.inflated                                       | surface | inflated cortical surface                                                                    |
| surf      | lh.pial, rh.pial                                               | surface | pial surface                                                                                 |
| surf      | lh.thickness, rh.thickness                                     | surface | cortical thickness overlay file                                                              |
| surf      | lh.volume, rh.volume                                           | surface | gray matter volume overlay file                                                              |
| surf      | lh.white, rh.white                                             | surface | white matter surface                                                                         |
| label     | lh.aparc.DKTatlas.annot, rh.aparc.DKTatlas.annot               | surface | symlink to lh.aparc.DKTatlas.mapped.annot                                                    |
| label     | lh.aparc.DKTatlas.mapped.annot, rh.aparc.DKTatlas.mapped.annot | surface | annotation file for cortical parcellations, mapped from ASEGDKT segmentation to the surface  |
| stats     | aseg.stats                                                     | surface | table of cortical and subcortical segmentation statistics after running the surface module   |
| stats     | lh.aparc.DKTatlas.mapped.stats, rh.aparc.DKTatlas.mapped.stats | surface | table of cortical parcellation statistics, mapped from ASEGDKT segmentation to the surface   |
| stats     | lh.curv.stats, rh.curv.stats                                   | surface | table of curvature statistics                                                                |
| stats     | wmparc.DKTatlas.mapped.stats                                   | surface | table of white matter segmentation statistics                                                |
| scripts   | recon-all.log                                                  | surface | logfile                                                                                      |