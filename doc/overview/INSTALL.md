Installation
============
FastSurfer works the same way on every system: you call `run_fastsurfer.sh` with a T1-weighted MRI image. How you
install it depends on your system. Pick yours:

````{grid} 1 2 2 2
:gutter: 3

```{grid-item-card} macOS
:link: install/MACOS
:link-type: doc

**Apple silicon:** the installer package, with all software included.

**Intel Macs:** Docker.
```

```{grid-item-card} Linux
:link: install/LINUX
:link-type: doc

**Apptainer/Singularity** or **Docker** images, for NVIDIA GPUs, AMD GPUs (experimental) or CPU only.
```

```{grid-item-card} Windows
:link: install/WINDOWS
:link-type: doc

**Docker** in WSL2, for NVIDIA GPUs or CPU only.
```

```{grid-item-card} From source
:link: install/NATIVE
:link-type: doc

A native installation on Ubuntu, for developers and systems without containers.
```
````

Which method should I use?
--------------------------

| Your system                          | Recommended                           | Alternatives                                              |
|--------------------------------------|---------------------------------------|-----------------------------------------------------------|
| Mac with Apple silicon (M1 or newer) | [macOS package][macos]                |                                                           |
| Mac with an Intel CPU                | [Docker][macos-docker]                |                                                           |
| Compute cluster (HPC)                | [Apptainer/Singularity][apptainer]    |                                                           |
| Linux workstation with NVIDIA GPU    | [Docker][docker]                      | [Apptainer/Singularity][apptainer], [from source][native] |
| Linux with AMD GPU                   | [Docker, ROCm build][amd]             |                                                           |
| Linux without GPU                    | [Apptainer or Docker, CPU image][cpu] |                                                           |
| Windows                              | [Docker in WSL2][windows]             |                                                           |

[macos]: install/MACOS.md
[macos-docker]: install/MACOS.md#docker-intel-macs
[apptainer]: install/LINUX.md#apptainer-or-singularity
[docker]: install/LINUX.md#docker
[amd]: install/LINUX.md#amd-gpus-experimental
[cpu]: install/LINUX.md#cpu-only
[windows]: install/WINDOWS.md
[native]: install/NATIVE.md

The containers and the macOS package include all software FastSurfer needs (only the
[FreeSurfer license](#freesurfer-license) for the surface reconstruction has to be obtained separately), and the
containers are what we test and validate FastSurfer with (Ubuntu
{{ UBUNTU_VERSION }}). A native installation depends on the software on your system, so its results can differ from
ours, and we may not be able to help if it does not work.

Before you start
----------------

### Hardware

A GPU makes the segmentation much faster. How much memory FastSurfer needs depends on the voxel size and on whether
the GPU or the CPU does the work; the [system requirements](intro.rst#system-requirements) list both.

### FreeSurfer license

The surface pipeline uses some FreeSurfer tools, so it needs a FreeSurfer license file, as does the Talairach
registration in the segmentation (`--tal_reg`, used for the estimated total intracranial volume, eTIV). A
segmentation without `--tal_reg` does not need one.

The license is free: [register at the FreeSurfer website](https://surfer.nmr.mgh.harvard.edu/registration.html) and
you receive it by email. Save the file in your home folder and pass it to FastSurfer with
`--fs_license <freesurfer_license_path>`, or set the `FS_LICENSE` environment variable to its path. A container also
needs access to the file, see the examples on the page of your system.

```{toctree}
:hidden:

install/MACOS.md
install/LINUX.md
install/WINDOWS.md
install/NATIVE.md
docker
SINGULARITY.md
```
