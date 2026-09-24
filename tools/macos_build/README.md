# FastSurfer macOS packaging
## Create macOS package

To build the macOS package of FastSurfer, run:

```bash
./build_release_package.sh arm
```

`arm` (arm64, Apple silicon) is the only buildable architecture. PyTorch publishes no macOS x86_64
wheels after 2.2, so the environment this package bundles cannot be resolved for Intel; an `intel`
invocation is refused. Intel Mac users run the Docker image instead, see
[INSTALL.md](../../doc/overview/INSTALL.md). Run the script without arguments for the full list of
options (cache directories for the FreeSurfer download, the pruned install, uv and the checkpoints).

### Dependencies for the script

[uv](https://docs.astral.sh/uv/) (`brew install uv`), which fetches the relocatable standalone Python
and the dependencies that get bundled into the package.

The FastSurfer applet is an AppleScript compiled with `osacompile`, and its icon is scaled from
`fastsurfer-icon.png` with `sips` and `iconutil`. All three ship with macOS, so the applet needs
nothing installed. `make_icon.py` regenerates that PNG from the logo, which needs Pillow.

### Running the package

After the script is executed, `installer` folder will be created along with the macOS package of FastSurfer inside.
Run the package by opening it and follow instructions.

After successful installation, FastSurfer applet and its source code will appear under the `/Applications` folder.

### Run FastSurfer

If you would want to run FastSurfer, you could either use terminal or FastSurfer applet. Though, running applet is recommended as it opens shell terminal and sets up environment for FastSurfer.

#### FastSurfer Flags
* The `--fs_license` points to your FreeSurfer license. 
* The `--t1` points to the t1-weighted MRI image to analyse (full, absolute path).
* The `--sid` is the subject ID name (folder name in output directory).
* The `--sd` points to the output directory.
* [more flags](../../doc/overview/FLAGS.md#fastsurfer-flags)

A directory with the name as specified in `--sid` (here subjectX) will be created in the output directory (specified via `--sd`). So in this example output will be written to /home/user/my_fastsurfer_analysis/subjectX/ . Make sure the output directory is empty, to avoid overwriting existing files. 

All other available flags are identical to the ones explained on the main page [README](../../README.md).
