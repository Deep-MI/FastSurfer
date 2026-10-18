FastSurfer runs in Docker, inside WSL2, the Linux environment of Windows:

1. Install WSL2: open PowerShell as administrator, run `wsl --install`, and restart Windows. This also installs
   Ubuntu.
2. Install [Docker Desktop](https://docs.docker.com/desktop/install/windows-install/), start it, and switch on the
   WSL integration for Ubuntu under **Settings > Resources > WSL integration**.
3. For an NVIDIA GPU, install the latest [NVIDIA driver for WSL](https://developer.nvidia.com/cuda/wsl) in Windows.
4. Open **Ubuntu** from the Start menu. Run the FastSurfer commands in this terminal, not in PowerShell. Check
   Docker there, for example:

   ```bash
   docker run --rm hello-world
   ```

WSL2 gets only part of the computer's memory; the [system requirements](../../intro.rst#system-requirements) say how
much FastSurfer needs, and the launcher warns if Docker has less.
