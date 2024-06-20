# Hypothalamus pipeline

Hypothalamic subfields segmentation pipeline

### Input
*  T1w image, a T2w image, or both images. Note: Input images to the tool need to be Bias-Field corrected.

### Requirements
* Same as FastSurfer.
* If the T1w and T2w images are available and not co-registered, FreeSurfer should be sourced to run the registration code, and the mri_coreg and mri_vol2vol binaries should also be available.

### Model weights
* EUDAT (FZ Jülich) data repository: https://b2share.fz-juelich.de/records/27ab0a28c11741558679c819d608f1e7
* Zenodo data repository: https://zenodo.org/records/10623893

### Pipeline Steps
1. Registration (optional, only required for multi-modal input)
2. Hypothalamus Segmentation

### Running the tool
Run the HypVINN/run_pipeline.py which has the following arguments:
### Input and output arguments
 * `--sid <name>` :  Subject ID, the subject data upon which to operate
 * `--sd <name>` : Directory in which evaluation results should be written.
 *  `--t1 </dir/T1**.nii.gz>` : T1 image path
 *  `--t2 </dir/T2**.nii.gz>` : T2 image path
 * `--seg_log` :  Path to file in which run logs will be saved. If not set logs will be stored in /sd/sid/logs/hypvinn_seg.log 
### Image processing options
 * `--no_reg` : Deactivate registration of T2 to T1. If multi-modal input is used; images need to be registered externally,
 * `--reg_mode` : Freesurfer Registration type to run. coreg : mri_coreg (Default) or robust : mri_robust_register.
 * `--qc_snap`: Activate the creation of QC snapshots of the predicted HypVINN segmentation.
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
The pipeline can do all pre-processing by itself (step 1). This step can be skipped if images are already registered externally. Note, that images are conformed as a first step, which can lead to additional interpolation reducing quality.

1. Run full pipeline
    ```
    python HypVINN/run_pipeline.py  --sid test_subject --sd /output \
                                     --t1 /data/test_subject_t1.nii.gz \
                                     --t2 /data/test_subject_t2.nii.gz \
                                     --reg_mode coreg \
                                     --seg_log /outdir/test_subject.log \
                                     --batch_size 6
   ```
2. Run full pipeline only using a t1 
    ```
    python HypVINN/run_pipeline.py  --sid test_subject --sd /output \
                                     --t1 /data/test_subject_t1.nii.gz \
                                     --reg_mode coreg \
                                     --seg_log /outdir/test_subject.log \
                                     --batch_size 6
   ```

3. Run pipeline without the registration step
    ```
    python HypVINN/run_pipeline.py  --sid test_subject --sd /output \
                                     --t1 /data/test_subject_t1.nii.gz \
                                     --t2 /data/test_subject_t2.nii.gz \
                                     --reg_mode coreg \
                                     --seg_log /outdir/test_subject.log \
                                     --batch_size 6 --no_reg
   ```

4. Run pipeline with creation of qc snapshots
    ```
    python HypVINN/run_pipeline.py  --sid test_subject --sd /output \
                                     --t1 /data/test_subject_t1.nii.gz \
                                     --t2 /data/test_subject_t2.nii.gz \
                                     --reg_mode coreg \
                                     --seg_log /outdir/test_subject.log \
                                     --batch_size 6 --qc_snap
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
        |-- qc_snapshots : QC outputs (optional)
            |-- hypothalamus.HypVINN_qc_screenshoot.png (Coronal quality control image)
        |-- stats : Statistics outputs                                                 
            |-- hypothalamus.HypVINN.stats (Segmentation stats)     
 ``` 


### Developer

Santiago Estrada : santiago.estrada@dzne.de

### Citation
If you use the HypVINN module please cite
```
Santiago Estrada, David Kügler, Emad Bahrami, Peng Xu, Dilshad Mousa, Monique M.B. Breteler, N. Ahmad Aziz, Martin Reuter; 
FastSurfer-HypVINN: Automated sub-segmentation of the hypothalamus and adjacent structures on high-resolutional brain MRI. 
Imaging Neuroscience 2023; 1 1–32. doi: https://doi.org/10.1162/imag_a_00034
```
