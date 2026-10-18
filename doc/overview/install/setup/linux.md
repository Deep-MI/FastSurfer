FastSurfer runs in a container, with Apptainer (or its predecessor Singularity) or Docker. Check whether one of them
works, for example:

```bash
# Apptainer (with Singularity: singularity --version)
apptainer --version
# Docker
docker run --rm hello-world
```

Clusters usually provide Apptainer, often as a module (`module load apptainer`, see your cluster's documentation).
Installing Apptainer or Docker needs administrator rights, so if neither works, ask your administrator to install
[Apptainer](https://apptainer.org/docs/admin/main/installation.html) or
[Docker](https://docs.docker.com/engine/install/). Docker also needs:

- your user in the `docker` group, to run containers without `sudo` (log out and back in once you were added),
- for NVIDIA GPUs, the
  [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
  Apptainer needs only the NVIDIA driver.
