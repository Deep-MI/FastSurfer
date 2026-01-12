FastSurfer Docker Support
=========================

Pull FastSurfer from DockerHub
------------------------------
We provide pre-built Docker images with support for nVidia GPU-acceleration and for CPU-only use on [Docker Hub](https://hub.docker.com/r/deepmi/fastsurfer/tags). In order to quickly get the latest Docker image, simply execute:

```bash 
docker pull deepmi/fastsurfer
```

This will download the newest, official FastSurfer image with support for nVidia GPUs.

Image are named and tagged as follows: `deepmi/fastsurfer:<support>-<version>`, where `<support>` is `gpu` for support of NVIDIA GPUs and `cpu` without hardware acceleration (the latter is smaller and thus faster to download).
Similarly, `<version>` can be a version string (`latest` or `v#.#.#`, where `#` are digits, for example `v2.5.0`), for example:

```bash 
docker pull deepmi/fastsurfer:cpu-v2.5.0
```

Running the (official) Docker Image
-----------------------------------
After pulling the image, you can start a FastSurfer container and process a T1-weighted image (both segmentation and 
surface reconstruction) with the following command:

```bash
docker run --gpus all -v $HOME/my_mri_data:/data \
                      -v $HOME/my_fastsurfer_analysis:/output \
                      -v $HOME/my_fs_license_dir:/fs_license \
                      --rm --user $(id -u):$(id -g) deepmi/fastsurfer:latest \
                      --fs_license /fs_license/license.txt \
                      --t1 /data/subjectX/t1-weighted.nii.gz \
                      --sid subjectX --sd /output \
                      --threads 4 --3T # and more flags
```

### Docker Flags
* `--gpus`: This argument is used to access GPU resources. With it, you can also specify how many GPUs to use. In the example above, `all` will make every GPU available to FastSurfer in the Docker container. To use a single one (e.g.  GPU 0), set `--gpus device=0`. To use multiple specific GPUs (e.g. GPU 0, 1 and 3), use `--gpus "device=0,1,3"`.
* `-v`: This argument defines which and how data is shared between the host system and the docker container. By default, no data is shared between the host and the container. `-v` is used to explicitly share data. It follows the format `-v <host folder>:<container folder>:<options>`. In its simplest form, `<host folder>` and `<container folder>` are the same and folders inside the container are the same as on the host. `:<options>` may be left out or `:ro` to indicate that files from this folder may not be modified by the docker container (readonly). The following files need to be shared: input files, output folder (subjects directory) and FreeSurfer license.
* `--user $(id -u):$(id -g)`: Which user the container runs as (relevant for file access, the user-id and group-id, **required**!). `$(id -u)` and `$(id -g)` determine the user and group, respectively. Running the docker container as root `--user 0:0` is strongly discouraged and must be combined with the FastSurfer flag `--allow_root`.
* `--rm`: The flag takes care of removing the container (cleanup of the container) once the analysis finished (optional, but recommended). 
* `-d`: You can add this flag to run in detached mode (no screen output, and you return to shell, optional).

#### Advanced Docker Flags
* `--group-add <list of groups>`: If additional user groups are required to access files, additional groups may be added via `--group-add <group id>[,...]` or `--group-add $(id -G <group name>)`. 

### FastSurfer Flags
In principle, the same as [basic run_fastsurfer.sh](../../doc/scripts/RUN_FASTSURFER.md#required-arguments) with the 
following modifications:
* The `--fs_license` cannot be auto-detected and must be passed. Must point to your FreeSurfer license how it is 
  accessible inside the container (check `-v` [above](#docker-flags), `--fs_license` works in concert with `-v`). 
* `--t1` and `--sd` are required! They work in concert with `-v` [above](#docker-flags).
* `--sid` is the subject ID name (output folder), same as in [basic run_fastsurfer.sh](../../doc/scripts/RUN_FASTSURFER.md).

A directory with the name as specified in `--sid` (here subjectX) will be created in the output directory (specified via `--sd`). So in this example output will be written to `$HOME/my_fastsurfer_analysis/subjectX/`. Make sure the output directory is empty, to avoid overwriting existing files. 

All other available flags are identical to the ones explained on the main page [README](../../README.md).

### Docker Best Practice
* Do not mount the user home directory into the docker container as the home directory.
  
  Why? If the user inside the docker container has access to a user directory, settings from that directory might bleed into the FastSurfer pipeline. For example, before FastSurfer 2.2 python packages installed in the user directory would replace those installed inside the image potentially causing incompatibilities. Since FastSurfer 2.2, `docker run ... --version +pip` outputs the FastSurfer version including a full list of python packages. 

  How? Docker does not mount the home directory by default, so unless you manually set the `HOME` environment variable, all should be fine. 

FastSurfer Docker Image Creation
--------------------------------
Within this directory, we currently provide a build script and Dockerfile to create multiple Docker images for users (usually developers) who wish to create their own Docker images for 3 platforms:

* Nvidia / CUDA (Example 1)
* CPU (Example 2)
* AMD / rocm (experimental, Example 3)

To run only the surface pipeline or only the segmentation pipeline, the entrypoint to these images has to be adapted, which is possible through
-  for the segmentation pipeline: `--entrypoint "python /fastsurfer/FastSurferCNN/run_prediction.py"`
-  for the surface pipeline: `--entrypoint "/fastsurfer/recon_surf/recon-surf.sh"`

Note, for many HPC users with limited GPUs or with very large datasets, it may be most efficient to run the full pipeline on the CPU, trading a longer run-time for the segmentation with massive parallelization on the subject level. 

Also note, in order to run our Docker containers on a Mac, users need to increase docker memory to 10 GB by overwriting the settings under Docker Desktop --> Preferences --> Resources --> Advanced (slide the bar under Memory to 10 GB; see: [Change Docker Desktop settings on Mac](https://docs.docker.com/desktop/settings/mac/) for details). For the new Apple silicon chips (M1,etc), we noticed that a native install runs much faster than docker, because the Apple Accelerator (use the experimental MPS device via `--device mps`) can be used. There is no support for MPS-based acceleration through docker at the moment. 

### General build settings
The build script `build.py` supports additional args, targets and options, see `python tools/Docker/build.py --help`.

Note, that the build script's main function is to select parameters for build args, but also create the FastSurfer-root/BUILD.info file, which will be used by FastSurfer to document the version (including git hash of the docker container). This BUILD.info file must exist for the docker build to be successful.
In general, if you specify `--dry_run` the command will not be executed but sent to stdout, so you can run `python build.py --device cuda --dry_run | bash` as well.

By default, the build script will tag your image as `"fastsurfer:[{device}-]{version_tag}"`, where `{version_tag}` is `{version-identifer from pyproject.toml}_{current git-hash}` and `{device}` is the value to `--device` (omitted for `cuda`), but a custom tag can be specified by `--tag {tag_name}`. 

#### BuildKit
Note, we recommend using BuildKit to build docker images (e.g. `DOCKER_BUILDKIT=1` -- the build.py script already always adds this). To install BuildKit, run `wget -qO ~/.docker/cli-plugins/docker-buildx https://github.com/docker/buildx/releases/download/<version>/buildx-<version>.<platform>`, for example `wget -qO ~/.docker/cli-plugins/docker-buildx https://github.com/docker/buildx/releases/download/v0.12.1/buildx-v0.12.1.linux-amd64`. See also https://github.com/docker/buildx#manual-download.

### Example 1: Build GPU FastSurfer Image
In order to build your own Docker image for FastSurfer (FastSurferCNN + recon-surf; on GPU; including FreeSurfer) yourself simply execute the following command after traversing into the *Docker* directory: 

```bash
python tools/Docker/build.py --device cuda --tag my_fastsurfer:cuda
```

The build script allows more specific options, that specify different CUDA options as well (see `build.py --help`).

For running the analysis, the command is the same as above for the prebuild option:
```bash
docker run --gpus all \
           -v $HOME/my_mri_data:$HOME/my_mri_data \
           -v $HOME/my_fastsurfer_analysis:$HOME/my_fastsurfer_analysis \
           -v $HOME/my_fs_license.txt:$HOME/my_fs_license.txt \
           --rm --user $(id -u):$(id -g) my_fastsurfer:cuda \
               --fs_license $HOME/my_fs_license.txt \
               --t1 $HOME/my_mri_data/subjectX/t1-weighted.nii.gz \
               --sid subjectX --sd $HOME/my_fastsurfer_analysis \
               --threads 4 --3T
```


### Example 2: Build CPU FastSurfer Image
In order to build the docker image for FastSurfer (FastSurferCNN + recon-surf; on CPU; including FreeSurfer) simply go to the parent directory (FastSurfer) and execute the docker build command directly:

```bash
python tools/Docker/build.py --device cpu --tag my_fastsurfer:cpu
```

As you can see, only the `--device` to the build command is changed from `cuda` to `cpu`. 

To run the analysis, the command is basically the same as above, except for removing the `--gpus all` GPU option:
```bash
docker run -v $HOME/my_mri_data:$HOME/my_mri_data \
           -v $HOME/my_fastsurfer_analysis:$HOME/my_fastsurfer_analysis \
           -v $HOME/my_fs_license.txt:$HOME/my_fs_license.txt \
           --rm --user $(id -u):$(id -g) my_fastsurfer:cpu \
               --fs_license $HOME/my_fs_license.txt \
               --t1 $HOME/my_mri_data/subjectX/t1-weighted.nii.gz \
               --device cpu \
               --sid subjectX --sd $HOME/my_fastsurfer_analysis \
               --threads 16 --3T
```

FastSurfer will automatically detect, that no GPU is available and use the CPU.

### Example 3: Experimental Build for AMD GPUs
Here we build an experimental image to test performance when running on AMD GPUs. Note that you need a supported OS and Kernel version and supported GPU for the RocM to work correctly. You need to install the Kernel drivers into your host machine kernel (`amdgpu-install --usecase=dkms`) for the amd docker to work. For this follow: https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/quick-start.html#rocm-install-quick, https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/amdgpu-install.html#amdgpu-install-dkms and https://rocm.docs.amd.com/projects/install-on-linux/en/latest/how-to/docker.html

```bash
python tools/Docker/build.py --device rocm --tag my_fastsurfer:rocm
```

and run segmentation only:

```bash
docker run --rm --security-opt seccomp=unconfined \
           --device=/dev/kfd --device=/dev/dri --group-add video \
	       -v $HOME/my_mri_data:$HOME/my_mri_data \
           -v $HOME/my_fastsurfer_analysis:$HOME/my_fastsurfer_analysis \
           -v $HOME/my_fs_license.txt:$HOME/my_fs_license.txt \
           --user $(id -u):$(id -g) my_fastsurfer:rocm \
               --fs_license $HOME/my_fs_license.txt \
               --t1 $HOME/my_mri_data/subjectX/t1-weighted.nii.gz \
               --sid subjectX --sd $HOME/my_fastsurfer_analysis \
               --parallel \
               # alternatively: --device cuda is also possible (or --device cuda:0 to specify the GPU
```

In conflict with the official ROCm documentation (above), we also needed to add the group render `--group-add render` (in addition to `--group-add video`).

Note, we tested on an AMD Radeon Pro W6600, which is [not officially supported](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html#supported-gpus), but setting `HSA_OVERRIDE_GFX_VERSION=10.3.0` [inside docker did the trick](https://en.opensuse.org/SDB:AMD_GPGPU#Using_CUDA_code_with_ZLUDA_and_ROCm):

```bash
docker run --rm --security-opt seccomp=unconfined \
           --device=/dev/kfd --device=/dev/dri --group-add video --group-add render \
	       -v $HOME/my_mri_data:$HOME/my_mri_data \
           -v $HOME/my_fastsurfer_analysis:$HOME/my_fastsurfer_analysis \
           -v $HOME/my_fs_license.txt:$HOME/my_fs_license.txt \
           -e HSA_OVERRIDE_GFX_VERSION=10.3.0 \
           --user $(id -u):$(id -g) my_fastsurfer:rocm \
               --fs_license $HOME/my_fs_license.txt \
               --t1 $HOME/my_mri_data/subjectX/t1-weighted.nii.gz \
               --sid subjectX --sd $HOME/my_fastsurfer_analysis \
               --parallel \
               # alternatively: --device cuda is also possible (or --device cuda:0 to specify the GPU
```

Build docker image with attestation and provenance
--------------------------------------------------
To build a docker image with attestation and provenance, i.e. Software Bill Of Materials (SBOM) information, several requirements have to be met:

1. The image must be built with version v0.11+ of BuildKit (we recommend you [install BuildKit](#buildkit) independent of attestation).
2. You must configure a docker-container builder in buildx (`docker buildx create --use --bootstrap --name fastsurfer-bctx --driver docker-container`). Here, you can add additional configuration options such as safe registries to the builder configuration (add `--config /etc/buildkitd.toml`).
   ```toml
   root = "/path/to/data/for/buildkit"
   [worker.containerd]
     gckeepstorage=9000
     [[worker.containerd.gcpolicy]]
       keepBytes = 512000000
       keepDuration = 172800
       filters = [ "type==source.local", "type==exec.cachemount", "type==source.git.checkout"]
     [[worker.containerd.gcpolicy]]
       all = true
       keepBytes = 1024000000
   ```
3. Attestation files are not supported by the standard docker image storage driver. Therefore, images cannot be tested locally. 
   There are two solutions to this limitation.
   1. Directly push to the registry: 
      Add `--action push` to the build script (the default is `--action load`, which loads the created image into the current docker context, and for the image name, also add the registry name. For example `... python tools/Docker/build.py ... --attest --action push --tag docker.io/<myaccount>/fastsurfer:latest`.
   2. [Install the containerd image storage driver](https://docs.docker.com/storage/containerd/#enable-containerd-image-store-on-docker-engine), which supports attestation: To implement this on Linux, make sure your docker daemon config file `/etc/docker/daemon.json` includes
      ```json
      {
          "features": {
              "containerd-snapshotter": true
          }
      }
      ```
      Also note, that the image storage location with containerd is not defined by the docker config file `/etc/docker/daemon.json`, but by the containerd config `/etc/containerd/config.toml`, which will likely not exist. You can [create a default config](https://github.com/containerd/containerd/blob/main/docs/getting-started.md#customizing-containerd) file with `containerd config default > /etc/containerd/config.toml`, in this config file edit the `"root"`-entry (default value is `/var/lib/containerd`).  
4. Finally, you can now build the FastSurfer image with `python tools/Docker/build.py ... --attest`. This will add the additional flags to the docker build command.

Building for release
--------------------
Make sure, you are building on a machine that has [containerd-storage and Buildkit](#build-docker-image-with-attestation-and-provenance).

```bash
# configuration
build_dir=$HOME/FastSurfer-build
img=deepmi/fastsurfer
# the version can be identified with: $build_dir/run_fastsurfer.sh --version
version=2.5.0
# the cuda and rocm version can be identified with: python $build_dir/tools/Docker/build.py --help | grep -E ^[[:space:]]+--device
cuda=128
cudas=("cuda118" "cuda126" "cuda$cuda")
rocm=6.3
rocms=("rocm$rocm")
# end of config

# code
git clone --branch stable --single-branch github.com/Deep-MI/FastSurfer $build_dir
cd $build_dir
all_tags=("latest" "gpu-latest" "cuda-v$version" "rocm-v$version" "cpu-latest")
# build all distinct images
for dev in cpu xpu "${rocms[@]}" "${cudas[@]}"
do
  python3 tools/Docker/build.py --tag $img:$dev-v$version --freesurfer_build_image $img-build:freesurfer741 --attest --device $dev
  all_tags+=("$dev-v$version")
done
# labels that are just references
docker tag $img:rocm$rocm-v$version $img:rocm-v$version
docker tag $img:cpu-v$version $img:cpu-latest
for tag in cuda-v$version gpu-latest latest; do docker tag $img:cu$cuda-v$version $img:$tag ; done
# push all labels
for tag in "${all_tags[@]}" ; do docker push $img:$tag ; done
```
