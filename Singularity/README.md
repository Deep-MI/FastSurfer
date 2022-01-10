# FastSurfer Singularity image creation

Within this directory we currently provide five different Singularity definiton files that are set up for running:

* the whole FastSurfer pipeline (FastSurferCNN + recon-surf, Example 1 (GPU) Example 2 (CPU))
* only the segmentation network (FastSurferCNN, Example 3 (GPU) Example 4 (CPU))
* only the surface pipeline (recon-surf, Example 5 (CPU))

In order to run the whole FastSurfer pipeline or the surface part, you need a valid FreeSurfer license (either from your local FreeSurfer installation or from the FreeSurfer website (https://surfer.nmr.mgh.harvard.edu/registration.html)).


## Set up Singularity

To build Singularity images localy you need to be root, but there is a workaround for running it for example from a compute cluster. You can remotly build your image at https://cloud.sylabs.io/builder. In order to use this feature you need an access token. 
For more in information see https://sylabs.io/guides/3.6/user-guide/ or see below in the FAQ.

## Initial setup

Make sure the FreeSurfer license is exported.

```bash
export FS_LICENSE=/pathToLicense/.license
```

NOTE: you do not need to export the FreeSurfer license when doing segmentation only (example 3 and 4)

### Example 1: Build GPU FastSurfer container

#### Build image:

```bash
singularity build --remote fastsurfer.sif fastsurfer.def
```
We assume we do not have root privileges so we are going to build as remote. 

__--remote__ : allows to build image without root privleges on a remote server. 


NOTE: 
* if you build your container as root you do not need the option __--remote__ and just write(singularity build fastsurfer.sif fastsurfer.def)  
* Singularity saves your .sif file on a server under https://cloud.sylabs.io/library/USER by default. So you need to check if you have enough space on the server. Otherwise you will get an Error.
To avoid this you can just delete the sif image on the server since you have it already in the folder in which you used the build command.

#### Run our Singualrity image.

```bash
export SINGULARITYENV_CUDA_VISIBLE_DEVICES=1

cd ..

singularity run --nv --bind /pathToLicense/.license,../my_mri_data/ \
				 ./Singularity/fastsurfer.sif \
				 --t1 ../my_mri_data/subject10/orig.mgz \
				 --sid subject10 \
				 --sd ../my_fastsurfer_analisis/ \
				 --parallel
```

NOTE: As default Singularity only binds the directories $HOME , /tmp , /proc , /sys , /dev, and $PWD. If your license or data is outside of this directories you will need to bind them with __--bind__. See the example below. You can also bind a location for the output if you want to save your outputs outside of the default binded directories.

We assume that our data and our licens is outside of he default directories. So we need to bind the locations first.
In this case we are binding the license and the data to our container.
We also want to use our GPU so we need to use __--nv__ to allow the container the use of the GPU. By default Singualrity uses all available GPUs to limit how many GPUs to use (for example one) you can set the flag __SINGULARITYENV_CUDA_VISIBLE_DEVICES=1__.

### Example 2: Build CPU FastSurfer container 

```bash
singularity build --remote fastsurfer_cpu.sif fastsurfer_cpu.def

cd ..

singularity run --bind /pathToLicense/.license \
				 ./Singularity/fastsurfer_cpu.sif \
				 --t1 ../my_mri_data/subject10/orig.mgz \
				 --sid subject10 \
				 --no_cuda
				 --sd ../my_fastsurfer_analisis/ \
				 --parallel
```

We assume that our license is outside of the default directories but the data lies somewhere in our /home dir. So we need to bind the locations first.
In this case we are binding only the license to our container.
We want to use our CPU so we do not need to use __--nv__ for singularity. But we still need specify for FastSurfer that we want to use out CPU, we can do this with the flag __--no_cuda__. 
### Example 3: Build GPU FastSurferCNN container (segmentation only)


```bash
singularity build --remote fastsurfer_cnn.sif fastsurfer_cnn.def

cd ..

singularity run --nv ./Singularity/fastsurfer_cnn.sif \
				 --i_dir ../my_mri_data/subject10/ \
				 --in_name orig.mgz \
				 --o_dir ../my_fastsurfer_analisis/ \
				 --out_name mri/aparc.DKTatlas+aseg.deep.mgz \
				 --log deep_surfer.log
```

We assume that our data lies somewhere in our /home dir. So we do not need to bind anything.
We want to use our GPU so we need __--nv__, this time we do not want to limit our GPU ressources so we do not set the SINGULARITYENV_CUDA_VISIBLE_DEVICES flag.
FreeSurfer is not needed so we do not need to export the license.

### Example 4: Build CPU FastSurferCNN container (segmentation only)


```bash
singularity build --remote fastsurfer_cnn_cpu.sif fastsurfer_cnn_cpu.def

cd ..

singularity run ./Singularity/fastsurfer_cnn.sif \
				 --i_dir ../my_mri_data/subject10/ \
				 --in_name orig.mgz \
				 --o_dir ../my_fastsurfer_analysis/ \
				 --out_name mri/aparc.DKTatlas+aseg.deep.mgz \
				 --no_cuda \
				 --log deep_surfer.log

```
Note: We do not need to export the FreeSurfer license path beause for segmentaton only we dont need FreeSurfer.

We assume that our data lies somewhere in our /home dir. So we need to bind the locations first.
In this case we do not need to bind anything.
We want to use our CPU so we do not need to use __--nv__ for singularity. But we still need specify for FastSurfer that we want to use out CPU, we can do this with the Flag __--no_cuda__. 
### Example 5: Build CPU FastSurfer recon-surf container (surface pipeline only)


```
singularity build --remote reconsurf.sif reconsurf.def

cd ..

singularity run --bind /pathToLicense/.license \  
				./Singularity/reconsurf.sif \
				--fs_license /pathToLicense/.license \
				--t1 ../my_mri_data/subject10/orig.mgz \     
				--sid subject10 \
				--sd ../my_fastsurfer_analysis \
				--parallel
```

We assume that our licens is outside of the default directories but the data lies somewhere in our /home dir. So we need to bind the locations first.
In this case we are binding only the license to our container.



## Frequent Problems:

* __Set up Singularity__
	
	#### Generate a acces token

     1. Go to: https://cloud.sylabs.io/
     2. Click “Sign In” and follow the sign in steps.
     3. Click on your login id (same and updated button as the Sign in one).
     4. Select “Access Tokens” from the drop down menu.
     5. Enter a name for your new access token, such as “test token”
     6. Click the “Create a New Access Token” button.
     7. Click “Copy token to Clipboard” from the “New API Token” page.
     8. Run singularity remote login and paste the access token at the prompt.

	For more in information see https://sylabs.io/guides/3.6/user-guide/ 
	
* __license not found__ 
	
	ERROR: FreeSurfer license file /opt/freesurfer/license.txt not found.

	you did not export the license:

	Answer:
	```
	export FS_LICENSE=/pathToLicense/.license
	```
	
* __remote builder storage is full__
	
	FATAL: Unable to push image to library: request did not succeed: quota error: storage quota exceeded (507 Insufficient Storage)
	
	You do not have suficcient space on the remote server.
	 
	Answer: eliminate images on https://cloud.sylabs.io/library/USER , and build again
	
	
* __wrong directory__ 
	
	ERROR: no such file or directory
 
	you might be in the wrong directory

	Answer: build in: ../FastSurfer/Singularity
			run in: ../FastSurfer
		
		
* __token expired__ 
	FATAL: While performing build: failed to post request to remote build service: Failed to verify auth token in request: token is expired (401 Unauthorized)
	   
	the token expired:
	   
	Answer: create an new Token (see Set up Singularity in FAQ)
 





