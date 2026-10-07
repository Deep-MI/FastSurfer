Windows
=======
On Windows, run FastSurfer in our Docker image, with Docker Desktop on WSL2. The image contains the full pipeline
(segmentation and surface reconstruction) and everything it needs.

Make sure that these are installed and running:
* [WSL2](https://learn.microsoft.com/en-us/windows/wsl/install)
* [Docker Desktop](https://docs.docker.com/desktop/install/windows-install/)

Then choose the image for your hardware. Run the commands in Windows PowerShell.

`````{tab-set}

````{tab-item} NVIDIA GPU
The GPU image also needs:
* Windows 11 or Windows 10 21H2 or newer,
* the latest WSL kernel, or at least 4.19.121 (5.10.16.3 or newer for better performance and fixes),
* an NVIDIA GPU and the latest [NVIDIA CUDA driver](https://developer.nvidia.com/cuda/wsl),
* the CUDA toolkit in WSL, see
  [CUDA Support for WSL 2](https://docs.nvidia.com/cuda/wsl-user-guide/index.html#cuda-support-for-wsl-2).

[Enable NVIDIA CUDA on WSL](https://learn.microsoft.com/en-us/windows/ai/directml/gpu-cuda-in-wsl) explains how to
install them. Then download the image:

```bash
docker pull deepmi/fastsurfer:{{ CUDA_STRING }}-v{{ FASTSURFER_VERSION }}
```

and run FastSurfer as in [Example 2](../EXAMPLES.md#example-2-fastsurfer-docker), for example:

```bash
docker run --gpus all --rm --user 1000:1000 \
    -v C:/Users/user/my_mri_data:/home/user/my_mri_data \
    -v C:/Users/user/my_fastsurfer_analysis:/home/user/my_fastsurfer_analysis \
    -v C:/path/to/your/freesurfer/license_file:/home/user/freesurfer_license \
    deepmi/fastsurfer:{{ CUDA_STRING }}-v{{ FASTSURFER_VERSION }} \
    --fs_license /home/user/freesurfer_license \
    --t1 /home/user/my_mri_data/subjectX/t1_weighted.nii.gz \
    --sid subjectX --sd /home/user/my_fastsurfer_analysis
```
````

````{tab-item} CPU only
Download the CPU image:

```bash
docker pull deepmi/fastsurfer:cpu-v{{ FASTSURFER_VERSION }}
```

and run FastSurfer as in [Example 2](../EXAMPLES.md#example-2-fastsurfer-docker), for example:

```bash
docker run --rm --user 1000:1000 \
    -v C:/Users/user/my_mri_data:/home/user/my_mri_data \
    -v C:/Users/user/my_fastsurfer_analysis:/home/user/my_fastsurfer_analysis \
    -v C:/path/to/your/freesurfer/license_file:/home/user/freesurfer_license \
    deepmi/fastsurfer:cpu-v{{ FASTSURFER_VERSION }} \
    --fs_license /home/user/freesurfer_license \
    --t1 /home/user/my_mri_data/subjectX/t1_weighted.nii.gz \
    --device cpu \
    --sid subjectX --sd /home/user/my_fastsurfer_analysis
```
````

`````

Windows paths cannot be mounted at the same path inside the container, so the examples map
`C:/Users/user/<dir_name>` to `/home/user/<dir_name>` and pass the container paths to FastSurfer. The full pipeline
needs a [FreeSurfer license](../INSTALL.md#freesurfer-license), mounted from where you saved it.

FastSurfer refuses to run as root or as the image's default user, so the examples pass a user with `--user`. On Linux
that is your own user (`$(id -u):$(id -g)`), but PowerShell has no `id` command, and Windows folders do not have
Linux user IDs anyway, so pass a fixed user ID instead, for example `1000:1000`.

Docker Desktop reserves only part of your memory for WSL2. FastSurfer needs at least the memory in the
[system requirements](../intro.rst#system-requirements); if it fails, check
[how much memory your WSL2 distribution has](https://www.aleksandrhovhannisyan.com/blog/limiting-memory-usage-in-wsl-2/).
