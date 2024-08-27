
# Manual Edits

## Pipeline

FreeSurfer allows the user to perform manual edits at various places in their pipeline.
A few of these can currently also be done in FastSurfer and we plan to add more in the future. 
However, we have noticed that much less editing is necessary as the neural net segmentation is
very robust. If users want to perform edits, it is important to understand that the order of
processing steps is different in FastSurfer:

1. We perform the segmentation in the *asegdkt module*. This step produces the segmentation file:
 - ```aparc.DKTatlas+aseg.deep.mgz```
2. Within the *asegdkt module* this segmentation is then also reduced to the aseg without CC and a brainmask:
 - ```aseg.auto_noCCseg.mgz```
 - ```mask.mgz```
3. Finally the *asegdkt module* computes a bias field corrected version of the conformed input image (needed to compute partial volume estimates for the stats file):
 - ```orig_nu.mgz```
4. After potentially other segmentation modules, the *surface module* (*recon-surf*) is run, which uses the above files as input.

## Possible Edits

Currently there is no way to interact with the surface module (stopping and resuming it).
Therefore, some typical FreeSurfer edits (such as brain finalsurfs) are not available.
Other edits (e.g. placement of WM control points) are not meaningful, as the segmentation is done differently with our neural network in the *asegdkt module*.

Therefore, currently three types of edits are possible:

## 1. T1 Pre-Processing: 
Instead of using the original scan as input, you can perform a bias field correction as a pre-processing step. This can also be achieved by running the *asegdkt module* twice (using different subject ids). The second time you input the bias field corrected image ```orig_nu.mgz```
that was provided from the first run. This can help brighten up some regions and improve segmentation quality for some difficult cases.

- Step 1: In first iteration of field correction method run full pipeline as follows: 
   ```bash
   # Source FreeSurfer
   export FREESURFER_HOME=/path/to/freesurfer
   source $FREESURFER_HOME/SetUpFreeSurfer.sh

   # Define data directory
   datadir=/home/user/my_mri_data
   fastsurferdir=/home/user/my_fastsurfer_analysis

   # Run FastSurfer
   ./run_fastsurfer.sh --t1 $datadir/subjectX/t1-weighted-nii.gz \
                    --sid subjectX --sd $fastsurferdir \
                    --parallel --threads 4 --3T
   ```
- Step 2: Run pipeline again for the second time, however this time input the bias field corrected image i.e ```orig_nu.mgz``` instead of original input image which was produced in first iteration. The file ```orig_nu.mgz``` can be found in output directory in mri subfolder. The output produced from the second iteration can be saved in a different output directory for comparative analysis with the output produced in first iteration.
   ```bash
    # Run FastSurfer
   ./run_fastsurfer.sh --t1 $datadir/subjectX/t1-weighted-nii.gz \
                    --sid subjectX --sd $fastsurferdir \
                    --parallel --threads 4 --3T
   ```

- Step 3: Run freeview or visualization 
   ```bash 
   freeview /path/to/output_directory/orig_nu.mgz
   ``` 
   Note: ```orig_nu.mgz``` file is not a segmented file, for segmentation load ```aparc.DKTatlas+aseg.deep.edited.mgz``` in freeview.


## 2. Segmentation Edits

You can manually edit ```aparc.DKTatlas+aseg.deep.mgz```. This is similar to aseg edits in FreeSurfer. You can fill-in undersegmented regions (with the correct segmentation ID). To re-create the aseg and mask run the following command before continuing with other modules:

- Step 1: Assuming that you have run the full fastsurfer pipeline once as described in method_1 and successfully produced segmentations and surfaces
- Step 2: Execute this command where reduce_to_aseg.py is located
   ```bash
   python3 reduce_to_aseg.py -i sid/mri/aparc.DKTatlas+aseg.deep.edited.mgz \ 
                             -o sid/mri/aseg.auto_noCCseg.mgz \
                             --outmask sid/mri/mask.mgz \
                             --fixwm
   ```
   Assuming you have edited ```aparc.DKTatlas+aseg.deep.edited.mgz``` in freeview, step_2 will produce two files i.e ```aseg.auto_noCCseg.mgz``` and ```mask.mgz ``` in the specified output folder. The output files can be loaded in freeview as a load volume. Edit-->load volume

- Step 3: For this step you would have to copy segmentation files produced in step_1, edited file ```aparc.DKTatlas+aseg.deep.edited.mgz``` and re-created file produced in step_2 in new output directory beforehand. 

   In this step you can then run surface module as follows:
   ```bash
   # Source FreeSurfer
   export FREESURFER_HOME=/path/to/freesurfer
   source $FREESURFER_HOME/SetUpFreeSurfer.sh

   # Define data directory
   datadir=/home/user/my_mri_data
   fastsurferdir=/home/user/my_fastsurfer_analysis

   # Run FastSurfer
   ./run_fastsurfer.sh --t1 $datadir/subjectX/t1-weighted-nii.gz \
                     --sid subjectX --sd $fastsurferdir \
                     --parallel --threads 4 \
                     --surf_only
   ```
   Note: ```t1-weighted-nii.gz``` would be the original input mri image.
   
   Do not forget to give fullpath to the input and output folder unless specified in variables.

## 3. Brainmask Edits: 
When surfaces go out too far, e.g. they grab dura, you can tighten the mask directly, just edit ```mask.mgz```and start the *surface module*. 

- Step 1: Assuming that you have run the full fastsurfer pipeline once as described in method_1 and successfully produced segmentations and surfaces
- Step 2: Edit ```mask.mgz``` file in freeview
- Step 3: Run the pipeline again in order to get the surfaces but before running the pipeline again do not forget to copy all the segmented files in to new input and output directory. 
   Note: The files in output folder should be pasted in the subjectX folder, the name of subjectX should be the same as it was used in step_1 otherwise it would raise an error of missing files even though the segmentation files exists in output folder.
   ```bash
   # Source FreeSurfer
   export FREESURFER_HOME=/path/to/freesurfer
   source $FREESURFER_HOME/SetUpFreeSurfer.sh

   # Define data directory
   datadir=/home/user/my_mri_data
   fastsurferdir=/home/user/my_fastsurfer_analysis

   # Run FastSurfer
   ./run_fastsurfer.sh --t1 $datadir/subjectX/t1-weighted-nii.gz \
                     --sid subjectX --sd $fastsurferdir \
                     --parallel --threads 4 \
                     --surf_only
   ```

   Note: ```t1-weighted-nii.gz``` would be the original input mri image.

   We hope that this will help with (some of) your editing needs. If more edits become available we will update this file. 
   Thanks for using FastSurfer. 
