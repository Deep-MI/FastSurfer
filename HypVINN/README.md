# Hypothalamus pipeline

Hypothalamic subfields segmentation pipeline

### Input
*  a T1w image, a T2w image, or both images

### Requirements
Same as FastSurfer
FreeSurfer should also be source to run the code and the mri_coreg ang mri_vol2vol binary should be also available

### Model weights
* EUDAT (FZ Jülich) data repository: https://b2share.fz-juelich.de/records/27ab0a28c11741558679c819d608f1e7
* Zenodo data repository: https://zenodo.org/records/10623893

### Pipeline Steps
1. Bias Field Correction Step (optional)
2. Registration (optional, only required for multi-modal input)
3. Hypothalamus Segmentation

### Running the tool
Run the HypVINN/run_pipeline.py which has the following arguments:
### Input and output arguments
 * `--sid <name>` :  Subject ID, the subject data upon which to operate
 * `--sd <name>` : Directory in which evaluation results should be written.
 *  `--t1 </dir/T1**.nii.gz>` : T1 image path, required = True
 *  `--t2 </dir/T2**.nii.gz>` : T2 image path, T2 image not required when running the t1 mode for the others is required , required = False
 * `--mode <name>` : Mode to run segmentation based on the available modalities. If is set to auto the model will choose the mode based on the passed input images, 
                     t1 : only T1 images, t2 : only T2 images or multi : both T1 and T2 images, default = 'auto'
 * `--seg_log` :  Path to file in which run logs will be saved. If not set logs will be stored in /sd/sid/logs/hypvinn_seg.log 
### Image processing options
 * `--no_pre_proc`: Deactivate all pre-processing steps, This is recommended when images are already bias field corrected and co-registered if T1 and T2 are available
 * `--no_bc` : Deactivate bias field correction, it is recommended to do bias field correction for calculating volumes taking account partial volume effects, reguired = False
 * `--no_reg` : Deactivate registration of T2 to T1. If multi mode is used; images need to be registered externally, required = False
 * `--reg_type` : Freesurfer Registration type to run. coreg : mri_coreg (Default) or robust : mri_robust_register.
###  FastSurfer Technical parameters (see FastSurfer documentation)
 * `--device`
 * `--viewgg_device`
 * `--threads`
 * `--batch_size`
 * `--async_io`
 * `--allow_root`

### Checkpoint to load
 * `--ckpt_cor </dir/to/coronal/ckpt>` : Coronal checkpoint to load, default =  $FASTSURFER_ROOT/checkpoints/HypVINN_axial_v1.0.0.pkl
 * `--ckpt_ax </dir/to/axial/ckpt>` : Axial checkpoint to load, default = $FASTSURFER_ROOT/checkpoints/HypVINN_coronal_v1.0.0.pkl
 * `--ckpt_sag </dir/to/sagittal/ckpt>` : Sagittal checkpoint to load, default = $FASTSURFER_ROOT/checkpoints/HypVINN_sagittal_v1.0.0.pkl

### CFG-file with default options for network
 * `--cfg_cor </dir/to/coronal/cfg>` : Coronal config file to load, default =  $FASTSURFER_ROOT/HypVINN/config/HypVINN_coronal_v1.0.0.yaml
 * `--cfg_ax </dir/to/axial/cfg>` : Axial config file to load, default =  $FASTSURFER_ROOT/HypVINN/config/HypVINN_axial_v1.0.0.yaml
 * `--cfg_sag </dir/to/sagittal/cfg>` : Sagittal config file to load, default =  $FASTSURFER_ROOT/HypVINN/config/HypVINN_sagittal_v1.0.0.yaml

### Usage
The Hypothalamus pipeline can be run by using a T1 a T2 or both images. 
Is recommended that all input images are bias field corrected and when passing both T1 and T2 they need to be co-registered.
The pipeline can do all pre-processing by itself (steps 1 and 2) or omit this step if images are already curated beforehand.

1. Run full pipeline
    ```
    python HypVINN/run_pipeline.py  --sid test_subject --sd /output \
                                     --t1 /data/test_subject_t1.nii.gz \
                                     --t2 /data/test_subject_t2.nii.gz \
                                     --reg_type coreg \
                                     --seg_log /outdir/test_subject.log \
                                     --batch_size 6
   ```
2. Run full pipeline t1 mode
    ```
    python HypVINN/run_pipeline.py  --sid test_subject --sd /output \
                                     --t1 /data/test_subject_t1.nii.gz \
                                     --reg_type coreg \
                                     --seg_log /outdir/test_subject.log \
                                     --batch_size 6 --mode t1
   ```

3. Run pipeline with no pre-processing
    ```
    python HypVINN/run_pipeline.py  --sid test_subject --sd /output \
                                     --t1 /data/test_subject_t1.nii.gz \
                                     --t2 /data/test_subject_t2.nii.gz \
                                     --reg_type coreg \
                                     --seg_log /outdir/test_subject.log \
                                     --batch_size 6 --no_pre_proc
   ```

4. Run pipeline with no bias field correction
    ```
    python HypVINN/run_pipeline.py  --sid test_subject --sd /output \
                                     --t1 /data/test_subject_t1.nii.gz \
                                     --t2 /data/test_subject_t2.nii.gz \
                                     --reg_type coreg \
                                     --seg_log /outdir/test_subject.log \
                                     --batch_size 6 --no_bc
   ```

### Output
```  bash
#Output Scheme 
|-- output_dir
    |--sid                                 
        |-- mri : MRI outputs
            |--hypothalamus.HypVINN.nii.gz(Hypothalamus Segmentation)
            |-- hypothalamus_mask.HypVINN.nii.gz (Hypothalamus Segmentation Mask)
            |-- transforms
                |-- t2tot1.lta (FreeSurfer registration file, only available if registration is performed)
        |-- qc_snapshots : QC outputs
            |-- hypothalamus.HypVINN_qc_screenshoot.png (Coronal quality control image)
        |-- stats : Statistics outputs                                                 
            |-- hypothalamus.HypVINN.stats (Segmentation stats)     
 ``` 

### Registration script
Registration of t1 and t2 is done by default using mri_coreg tool with the following commands
```
mri_coreg --mov /path/to/t1_image --targ path/to/t2_image --reg /outdir/test_subject/transforms/t2tot1.lta

mri_vol2vol --mov /path/to/t1_image --targ path/to/t2_image --reg /outdir/test_subject/transforms/t2tot1.lta --o /outdir/test_subject/mri/T2_reg.nii.gz --cubic --keep-precision
```

### Developer

Santiago Estrada : santiago.estrada@dzne.de
