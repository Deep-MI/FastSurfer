From source (native installation)
=================================
A native installation runs FastSurfer directly on your system, without a container. It is the setup for developers
and for systems where containers are not available. You install all dependencies yourself (system packages, Python
packages and FreeSurfer in the supported version), so the results can differ from our testing environment, and we
may not be able to help if something does not work. We test FastSurfer {{ FASTSURFER_VERSION }} with Ubuntu
{{ UBUNTU_VERSION }}, the base of our Docker images, and the steps below are for Ubuntu.

1. System packages
------------------
You need a few packages that may be missing on your system (this needs sudo access, or ask a system admin):

```bash
sudo apt-get update && sudo apt-get install -y --no-install-recommends \
      wget \
      git \
      ca-certificates \
      file
```

You also need bash 3.2 or newer (check with `bash --version`). These packages are enough to install the Python
dependencies and run the segmentation. The full pipeline also needs FreeSurfer (step 5).

2. uv for Python
----------------
We recommend [uv](https://docs.astral.sh/uv/) to manage the Python environment and packages. It is very fast and
makes managing different environments easy. See
[uv's documentation](https://docs.astral.sh/uv/getting-started/installation/) for more on installing it, for example
[shell autocompletion](https://docs.astral.sh/uv/getting-started/installation/#shell-autocompletion).

```bash
wget -qO- https://astral.sh/uv/install.sh | sh
```

3. FastSurfer
-------------
Get FastSurfer from GitHub. Choose the `stable` branch (tested thoroughly) or the `dev` branch (newest, but it can
be broken). For example, `stable`:

```bash
export FASTSURFER_HOME=${FASTSURFER_HOME:-/path/to/FastSurfer}
# FastSurfer will get cloned to $FASTSURFER_HOME
git clone --branch stable https://github.com/Deep-MI/FastSurfer.git \
    $FASTSURFER_HOME
cd $FASTSURFER_HOME
```

4. Python environment
---------------------
Create a new environment and install the FastSurfer dependencies:

```bash
# make sure you are in the FastSurfer directory!
# create a .venv environment directory inside the FastSurfer directory,
# e.g., python {{ PYTHON_VERSION }} (recommended)
uv venv --python python{{ PYTHON_VERSION }}
# install packages with pinned versions from the last stable release
# (recommended, that is what we tested with)
# uv pip sync only runs if uv pip compile succeeds
resolved=$(uv pip compile --no-build --torch-backend auto requirements.txt) && \
    uv pip sync --no-build --torch-backend auto - <<< "$resolved"
```

To select the PyTorch backend yourself, for example for testing, replace `auto` in both commands, for example with
`cpu` or `{{ CUDA_STRING }}`:

```bash
# make sure you are in the FastSurfer directory!
resolved=$(uv pip compile --no-build --torch-backend cpu requirements.txt) && \
    uv pip sync --no-build --torch-backend cpu - <<< "$resolved"
```

> **For developers:** To install the newest compatible dependency versions instead of the pinned stable ones, resolve
> from `pyproject.toml`, optionally with extras such as `--extra doc` or `--extra all`:
> ```bash
> resolved=$(uv pip compile --torch-backend auto --extra doc pyproject.toml) && \
>     uv pip sync --torch-backend auto - <<< "$resolved"
> ```
> Leave out `--no-build` here: `bibtexparser`, which the `style` extra needs, only ships source code (pure Python,
> no compiler needed), and `--no-build` makes `uv` resolve different versions to avoid it. `bibtexparser` should be
> the only package `uv` builds (`Building bibtexparser`).

Activate the FastSurfer environment with:

```bash
source .venv/bin/activate
```

and add the FastSurfer directory to the Python path:

```bash
# make sure you are in the FastSurfer directory!
export PYTHONPATH="${PYTHONPATH}:$PWD"
```

You need to do this every time you run FastSurfer, or add the line to your `~/.bashrc` if you use bash, for example:

```bash
# make sure you are in the FastSurfer directory!
echo "export PYTHONPATH=\"\${PYTHONPATH}:$(pwd)\"" >> ~/.bashrc
```

You can also download all network checkpoint files now (do this if you install for several users):

```bash
export FASTSURFER_HOME=${FASTSURFER_HOME:-/path/to/FastSurfer}
python3 $FASTSURFER_HOME/FastSurferCNN/download_checkpoints.py --all
```

With this, the segmentation runs (`run_fastsurfer.sh --seg_only ...`), see
[Example 3](../EXAMPLES.md#example-3-native-fastsurfer-on-subjectx-with-parallel-processing-of-hemis) for the
command line flags.

5. FreeSurfer
-------------
The full pipeline needs FreeSurfer {{ FREESURFER_VERSION }} (the version we recommend and support), installed
according to [FreeSurfer's instructions](https://surfer.nmr.mgh.harvard.edu/fswiki/DownloadAndInstall). The
packages for each version and operating system are in the
[release directory](https://surfer.nmr.mgh.harvard.edu/pub/dist/freesurfer/). If you run into problems in this
step, the FreeSurfer mailing list can help.

FastSurfer runs FreeSurfer's Talairach registration (`talairach_avi` and the tools it calls), which some FreeSurfer
packages leave out, among them FreeSurfer 8's packages for Ubuntu. Use a package that includes these tools, for
example the one for Rocky Linux, or run the full pipeline with our Docker or Singularity image. FastSurfer checks for
`talairach_avi` before it starts and stops with an error if it is missing.

Set the `FREESURFER_HOME` environment variable, so FastSurfer finds the FreeSurfer programs, and have a
[FreeSurfer license](../INSTALL.md#freesurfer-license).
