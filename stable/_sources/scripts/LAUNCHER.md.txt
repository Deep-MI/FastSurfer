fastsurfer (launcher)
=====================
`fastsurfer` runs FastSurfer with one command on every system: in its container image with Apptainer, Singularity or
Docker, or on a Mac with Apple silicon with the macOS package. It takes the flags of
[run_fastsurfer.sh](RUN_FASTSURFER.md) and needs only Python, not FastSurfer itself.
[The launcher](../overview/CONTAINERS.md#the-launcher) explains how it chooses the image, and the
[quick start](../overview/QUICKSTART.md) how to download it.

Usage
-----
```{command-output} python3 tools/fastsurfer --help
:cwd: /../
```
