
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

Currently there is no way to interact with the surface module (stoping and resuming it).
Therefore, some typical FreeSurfer edits (such as brain finalsurfs) are not available.
Other edits (e.g. placement of WM control points) are not meaningful, as the segmentation is done differently with our neural network in the *asegdkt module*.

Therefore, currently three types of edits are possible:

1. Instead of using the original scan as input, you can perform a bias field correction as a pre-processing step. This can also be achieved by running the *asegdkt module* twice (using different subject ids). The second time you input the bias field corrected image ```orig_nu.mgz``` that was provided from the first run. This can help brighten up some regions and improve segmentation quality for some difficult cases.
2. You can manually edit ```aparc.DKTatlas+aseg.deep.mgz```. This is similar to aseg edits in FreeSurfer. You can fill-in undersegmented regions (with the correct segmentation ID). To re-create the aseg and mask run the following command before continuing with other modules:

   ```
   python3 reduce_to_aseg.py -i sid/mri/aparc.DKTatlas+aseg.deep.edited.mgz 
                             -o sid/mri/aseg.auto_noCCseg.mgz 
                             -outmask sid/mri/mask.mgz 
                             --fixwm
   ```
3. When surfaces go out too far, e.g. they grab dura, you can tighten the mask directly, just edit ```mask.mgz```and start the *surface module*. 

We hope that this will help with (some of) your editing needs. If more edits become availble we will update this file. 
Thanks for using FastSurfer. 

