# FastSurfer MacOS packaging
## Create MacOS package

In order to build the MacOS package of FastSurfer, simply run:

```bash
./build_release_package.sh <version> <architecture> <path_to_fastsurfer_to_package> [<link_to_freesurfer>]
```

Script creates release package for MacOS, where `<version>` is the release version, `<architecture>` is `arm` for arm64 arch based chips and `intel` for `x86_64` arch based chips.
`<path_to_fastsurfer_to_package>` is the directory with the fastsurfer to package.
Link to specific freesurfer distribution might be provided with `<link_to_freesurfer>` argument.

### Dependencies for the script

Script is using py2app python dependency, which isn't installed through any requirements file of FastSurfer, so in order to run the script, make sure that py2app is installed.

### Running the package

After the script is executed, `installer` folder will be created along with the MacOS package of FastSurfer inside.
Run the package by opening it and follow instructions.

After successful installation, FastSurfer applet and its source code will appear under the `/Applications` folder.

### Run FastSurfer

If you would want to run FastSurfer, you could either use terminal or FastSurfer applet. Though, running applet is recommended as it opens shell terminal and sets up environment for FastSurfer.

#### FastSurfer Flags
* The `--fs_license` points to your FreeSurfer license which needs to be available on your computer in the `my_fs_license_dir` that was mapped above. 
* The `--t1` points to the t1-weighted MRI image to analyse (full path, with mounted name inside docker: /home/user/my_mri_data => /data)
* The `--sid` is the subject ID name (output folder name)
* The `--sd` points to the output directory (its mounted name inside docker: /home/user/my_fastsurfer_analysis => /output)
* [more flags](../../doc/overview/FLAGS.md#fastsurfer-flags)

Note, that the paths following `--fs_license`, `--t1`, and `--sd` are __inside__ the container, not global paths on your system, so they should point to the places where you mapped these paths above with the `-v` arguments. 

A directory with the name as specified in `--sid` (here subjectX) will be created in the output directory (specified via `--sd`). So in this example output will be written to /home/user/my_fastsurfer_analysis/subjectX/ . Make sure the output directory is empty, to avoid overwriting existing files. 

All other available flags are identical to the ones explained on the main page [README](../../README.md).
