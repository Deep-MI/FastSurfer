# FastSurfer Singularity image creation

Within this directory we currently provide five different Singularity definiton files that are set up for running:

* the whole FastSurfer pipeline (FastSurferCNN + recon-surf, Example 1 (GPU) Example 2 (CPU))
* only the segmentation network (FastSurferCNN, Example 3 (GPU) Example 4 (CPU))
* only the surface pipeline (recon-surf, Example 5 (CPU))

In order to run the whole FastSurfer pipeline or the surface part, you need a valid FreeSurfer license (either from your local FreeSurfer installation or from the FreeSurfer website (https://surfer.nmr.mgh.harvard.edu/registration.html)).


## Set up Singularity

To build Singularity images localy you need to be root, but there is a workaround for running it for example from a compute cluster. You can remotly build your image at https://cloud.sylabs.io/builder. In order to use this feature you need an access token. 

 
### Generate a acces token

     1. Go to: https://cloud.sylabs.io/
     2. Click “Sign In” and follow the sign in steps.
     3. Click on your login id (same and updated button as the Sign in one).
     4. Select “Access Tokens” from the drop down menu.
     5. Enter a name for your new access token, such as “test token”
     6. Click the “Create a New Access Token” button.
     7. Click “Copy token to Clipboard” from the “New API Token” page.
     8. Run singularity remote login and paste the access token at the prompt.

For more in information see https://sylabs.io/guides/3.6/user-guide/ 

Inorder to avoid overflow errors you should set the __SINGULARITY_TMPDIR__ and __SINGULARITY_CACHEDIR__ to be saved at a location with sufficient space. 

```bash
mkdir ~/.singularity/tmp
export SINGULARITY_TMPDIR=~/.singularity/tmp
mkdir ~/.singularity/cache
export SINGULARITY_CACHEDIR=~/.singularity/cache
```

NOTE: you need to do this every time u open a new terminal. Otherwiese you can write this setting in your ~/.bashrc (after you created the dedicated folders):

```bash
export SINGULARITY_TMPDIR=~/.singularity/tmp
export SINGULARITY_CACHEDIR=~/.singularity/cache
```


### Example 1: Build GPU FastSurfer container

Define the location where the tmp_data of the images should be located.

```bash
mkdir ~/.singularity/tmp
export SINGULARITY_TMPDIR=~/.singularity/tmp
mkdir ~/.singularity/cache
export SINGULARITY_CACHEDIR=~/.singularity/cache
```

#### Now we can build our Singualrity image.

We assume we do not have root privileges so we are going to build as remote. 

__--remote__ : allows to build image without root privleges on a remote server. 

```bash
singularity build --remote fastsurfer.sif fastsurfer.def
```

Note: * if you build your container as root you do not need the option __--remote__ and just write(singularity build fastsurfer.sif fastsurfer.def)  
      * Singularity saves your .sif file on a server under https://cloud.sylabs.io/library/USER by default. So you need to check if you have enough space on the server. Otherwise you will get an Error.
To avoid this you can just delete the sif image on the server since you have it already in the folder in which you used the build command.

#### Run our Singualrity image.

NOTE: As default Singularity only binds the directories $HOME , /tmp , /proc , /sys , /dev, and $PWD. If your license or data is outside of this directories you will need to bind them with __--bind__. See the example below.

We assume that our data and our licens is outside of he default directories. So we need to bind the locations first.
In this case we are binding the license and the data to our container.
We also want to use our GPU so we need to use __--nv__ to allow the container the use of the GPU. By default Singualrity uses all available GPUs to limit how many GPUs to use (for example one) you can set the flag __SINGULARITYENV_CUDA_VISIBLE_DEVICES=1__.


We first need to export our FreeSurfer license path. 

```bash
export FS_LICENSE=/pathToLicense/.license
```

Set using one GPU:
```bash
export SINGULARITYENV_CUDA_VISIBLE_DEVICES=1
```

```bash
cd ..

singularity run --nv --bind /pathToLicense/.license,/home/user/my_mri_data/ \
				 ./Singularity/fastsurfer.sif \
				 --t1 /home/user/my_mri_data/subject10/orig.mgz \
				 --sid subject10 \
				 --sd /home/user/my_fastsurfer_analisis/ \
				 --parallel
```
	
### Example 2: Build CPU FastSurfer container 

Define the location where the tmp_data of the images should be located.

```bash
export SINGULARITY_TMPDIR=~/.singularity/tmp
export SINGULARITY_CACHEDIR=~/.singularity/cache
```
Now we can build our image:

```bash
singularity build --remote fastsurfer_cpu.sif fastsurfer_cpu.def
```

We assume that our license is outside of the default directories but the data lies somewhere in our /home dir. So we need to bind the locations first.
In this case we are binding only the license to our container.
We want to use our CPU so we do not need to use __--nv__ for singularity. But we still need specify for FastSurfer that we want to use out CPU, we can do this with the flag __--no_cuda__. 

We first need to export the Freesurfer license.

```bash
export FS_LICENSE=/pathToLicense/.license
```

```bash
cd ..


singularity run --bind /pathToLicense/.license \
				 ./Singularity/fastsurfer_cpu.sif \
				 --t1 /home/user/my_mri_data/subject10/orig.mgz \
				 --sid subject10 \
				 --no_cuda
				 --sd /home/user/my_fastsurfer_analisis/ \
				 --parallel
```

### Example 3: Build GPU FastSurferCNN container (segmentation only)

Define the location where the tmp_data of the images should be located.


```bash
export SINGULARITY_TMPDIR=~/.singularity/tmp
export SINGULARITY_CACHEDIR=~/.singularity/cache
```

#### Build image:

```bash
singularity build --remote fastsurfer_cnn.sif fastsurfer_cnn.def
```

#### Run fastsurefer_cnn.sif:
We assume that our data lies somewhere in our /home dir. So we do not need to bind anything.
We want to use our GPU so we need __--nv__, this time we do not want to limit our GPU ressources so we do not set the SINGULARITYENV_CUDA_VISIBLE_DEVICES flag.
FreeSurfer is not needed so we do not need to export the license.

```bash
cd ..


singularity run --nv ./Singularity/fastsurfer_cnn.sif \
				 --i_dir /home/user/my_mri_data/subject10/ \
				 --in_name orig.mgz \
				 --o_dir /home/user/my_fastsurfer_analisis/ \
				 --out_name mri/aparc.DKTatlas+aseg.deep.mgz \
				 --log deep_surfer.log
```

### Example 4: Build CPU FastSurferCNN container (segmentation only)

Define the location where the tmp_data of the images should be located.

```
export SINGULARITY_TMPDIR=~/.singularity/tmp
export SINGULARITY_CACHEDIR=~/.singularity/cache
```


Now we can build our image:

```bash
singularity build --remote fastsurfer_cnn_cpu.sif fastsurfer_cnn_cpu.def
```

We do not need to export the Freesurfer license path beause for segmentaton only we dont need freesurfer.

We assume that our data lies somewhere in our /home dir. So we need to bind the locations first.
In this case we do not need to bind anything.
We want to use our CPU so we do not need to use __--nv__ for singularity. But we still need specify for FastSurfer that we want to use out CPU, we can do this with the Flag __--no_cuda__. 

```bash
cd ..

singularity run ./Singularity/fastsurfer_cnn.sif \
				 --i_dir /home/user/my_mri_data/subject10/ \
				 --in_name orig.mgz \
				 --o_dir /home/user/my_fastsurfer_analysis/ \
				 --out_name mri/aparc.DKTatlas+aseg.deep.mgz \
				 --no_cuda \
				 --log deep_surfer.log

```
### Example 5: Build CPU FastSurfer recon-surf container (surface pipeline only)

#### Build image:

```
singularity build --remote reconsurf.sif reconsurf.def
```
We assume that our licens is outside of he default directories but the data lies somewhere in our /home dir. So we need to bind the locations first.
In this case we are binding only the license to our container.
We can also bind to a specific destination ( __--bind /src:/dest__ ) see below:
```bash
cd ..

singularity run --bind /pathToLicense/.license:/license     
				./Singularity/reconsurf.sif \
				--fs_license /license \
				--t1 /home/user/my_mri_data/subject10/orig.mgz \     
				--sid subject10 \
				--sd /home/user/my_fastsurfer_analysis \
				--parallel
```

### Frequent Problems:

* Did you export the license ?
```
export FS_LICENSE=/pathToLicense/.license
```
* Space on Server ? (FATAL:   Unable to push image to library: request did not succeed: quota error: storage quota exceeded (507 Insufficient Storage))

 eliminate images on https://cloud.sylabs.io/library/USER , and build again
	
* Did you define a locale location to save the tmp_data ?
```
export SINGULARITY_TMPDIR=~/.singularity/tmp
export SINGULARITY_CACHEDIR=~/.singularity/cache
```
	
* are you in the correct directory ?
 build in ../FastSurfer/Singularity
 run in ../FastSurfer
	
	


