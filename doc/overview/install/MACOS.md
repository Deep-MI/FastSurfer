macOS
=====
On a Mac with Apple silicon (M1 or newer), install the FastSurfer package: it contains all software FastSurfer needs,
and it uses the Apple GPU automatically. The surface reconstruction also needs a [FreeSurfer license](#freesurfer-license),
which is free but not included. On a Mac with an Intel CPU, use [Docker](#docker-intel-macs) instead.

````{card}
:class-card: sd-border-primary sd-shadow-sm
:text-align: center

**FastSurfer for macOS**
^^^
Apple silicon (M1 or newer) · macOS 14 (Sonoma) or newer · about 1 GB download

Includes its own Python, all Python packages, the network checkpoints and the FreeSurfer tools FastSurfer uses:
no other software to install, and no internet connection needed after the download.

```{button-link} https://github.com/Deep-MI/FastSurfer/releases/latest/download/FastSurfer-macos-darwin_arm64.pkg
:color: primary
:shadow:

{octicon}`download` Download the installer (.pkg)
```
+++
[Other versions](https://github.com/Deep-MI/FastSurfer/releases/) · [Intel Mac? Use Docker](#docker-intel-macs)
````

Install
-------
1. Download the installer with the button above. It always points to the newest release.
2. Double-click the downloaded `.pkg` file and follow the installer. You need about 2.5 GB of free disk space.
3. The installer places two items in your Applications folder: `FastSurfer<version>`, the installation, and
   `FastSurfer<version>.app`, the app that starts FastSurfer.

````{dropdown} macOS blocks the installer ("cannot be opened")
:icon: shield-lock

The installer is not signed by Apple yet, so macOS Gatekeeper blocks it. Depending on your macOS version, the
warning may not offer an "Open" button at all, only "Done" or "Move to Trash". To allow it:

1. Click **Done** on the warning.
2. Open **System Settings > Privacy & Security** and scroll down to the **Security** section.
3. Click **Open Anyway** next to the message about the blocked installer, and confirm once more (with your password
   or Touch ID).
4. Double-click the `.pkg` file again to start the installation.
````

Run FastSurfer
--------------
Start the FastSurfer app from your Applications folder (or with Spotlight). It opens a Terminal window with a
FastSurfer console, recognizable by the `(FastSurfer<version>)` prompt, where everything is set up to run
FastSurfer. The first time, macOS asks whether FastSurfer may control Terminal; allow it, because that is how the app
opens the console. If you declined, switch on **Terminal** below **FastSurfer** in
**System Settings > Privacy & Security > Automation**.

In the console, call `run_fastsurfer.sh` with the [FastSurfer flags](../../scripts/RUN_FASTSURFER.md). For example,
the segmentation only:

```bash
run_fastsurfer.sh --seg_only --sd $HOME/my_fastsurfer_analysis --sid subjectX \
    --t1 $HOME/my_mri_data/subjectX/t1_weighted.nii.gz
```

For the full pipeline with surfaces, pass your FreeSurfer license, for example:

```bash
freesurfer_license=${freesurfer_license:-/path/to/your/freesurfer/license_file}
run_fastsurfer.sh --sd $HOME/my_fastsurfer_analysis --sid subjectX \
    --t1 $HOME/my_mri_data/subjectX/t1_weighted.nii.gz \
    --fs_license $freesurfer_license
```

FastSurfer uses the Apple GPU (`mps`) automatically, so no `--device` flag is needed. A few operations have no GPU
implementation yet and run on the CPU instead; PyTorch prints a warning naming them, which is harmless.

FreeSurfer license
------------------
On macOS, FastSurfer cannot find the [FreeSurfer license](../INSTALL.md#freesurfer-license) inside the bundled
FreeSurfer (`$FREESURFER_HOME`), because the installer places that folder as `root` and you cannot copy files into
it. Save the license file in your home folder instead, and either pass it with `--fs_license` as above or set it
for the current console:

```bash
export FS_LICENSE=/path/to/your/freesurfer/license_file
```

To set it in every console, add that line to your shell profile yourself (`~/.zprofile` for zsh, the macOS default,
or `~/.bash_profile` for bash). FastSurfer does not change these files.

More details
------------

````{dropdown} What the FastSurfer console sets up
The console is a bash session that:
- puts the Python bundled with FastSurfer (`$FASTSURFER_HOME/python`) first on `PATH`,
- sets `FASTSURFER_HOME` and `PYTHONPATH`,
- sets `FREESURFER_HOME` to the FreeSurfer tools bundled with FastSurfer and sources `SetUpFreeSurfer.sh`,
- adds the FastSurfer folder (and GNU `grep`, if you have it from Homebrew) to `PATH`, for this session only,
- reads your `~/.bashrc` first, if you have one, so your own aliases and settings are still there, and
- reminds you to set `FS_LICENSE` if it is not set yet.

No shell profile is modified.
````

````{dropdown} Using FastSurfer without the app
In a bash or zsh Terminal window, you can set up the same environment by sourcing the script the app uses:

```bash
source /Applications/FastSurfer{{ FASTSURFER_VERSION }}/macos_setup_fastsurfer.sh
```

The script is bash syntax, so from tcsh or fish, start `bash` (or `zsh`) first and source it there.

Only adding the FastSurfer folder to your `PATH` is not enough: `run_fastsurfer.sh` would be found, but `python3`
would be Apple's system Python, which is too old for FastSurfer, and `FREESURFER_HOME` would not be set.
````

````{dropdown} Requirements in detail
- **Apple silicon only:** PyTorch publishes no macOS packages for Intel CPUs, so the Python environment the package
  bundles cannot be built for them. On an Intel Mac, use [Docker](#docker-intel-macs).
- **macOS 14 (Sonoma) or newer:** the bundled programs are built for it and do not start on older versions. We do not
  test specific macOS versions, so this is a lower bound rather than a support statement.
- **Shells:** only the shells macOS ships, `/bin/bash` for FastSurfer's scripts and `/bin/tcsh` for FreeSurfer's.
  Your own Terminal shell does not matter, because the app starts a bash session itself.
- **No Python and no Homebrew** are needed.
````

````{dropdown} Uninstalling
Drag both items from your Applications folder to the Trash:
- `FastSurfer<version>` (the installation)
- `FastSurfer<version>.app` (the app)

macOS asks for your password, because the installer placed them as `root`. Everything FastSurfer installed is in
these two items: no shell profile is modified and nothing is written elsewhere. Installations of other versions are
independent and stay as they are.

To also remove the installer's receipt (bookkeeping only, it does not affect anything you run):

```text
sudo pkgutil --forget org.deep-mi.FastSurfer.<version_without_dots>_<arch>
```

`pkgutil --pkgs | grep -i fastsurfer` lists the exact names.
````

Docker (Intel Macs)
-------------------
The package does not run on Intel Macs, but Docker does, with the full pipeline on the CPU (2 to 4 times slower
than on Apple silicon).

1. Install [Docker Desktop for Mac](https://docs.docker.com/get-docker/), start it, and under
   **Settings > Resources** set the memory to 15 GB (or the most you have; with less than that, it may fail).
2. Download the CPU image. Open a Terminal window and run:

   ```bash
   docker pull deepmi/fastsurfer:cpu-v{{ FASTSURFER_VERSION }}
   ```

3. Run FastSurfer as in [Example 2](../EXAMPLES.md#example-2-fastsurfer-docker), with the CPU image and without
   `--gpus all`.
