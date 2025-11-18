# FastSurfer MacOS packaging
## Create MacOS package

In order to build the MacOS package of FastSurfer, simply run:

```bash
./build_release_package.sh <architecture>
```

Script creates release package for MacOS, where `<architecture>` is `arm` for arm64 arch based chips and `intel` for `x86_64` arch based chips.

### Dependencies for the script

Script is using py2app python dependency, which isn't installed through any requirements file of FastSurfer, so in order to run the script, make sure that py2app is installed.

### Running the package

After the script is executed, `installer` folder will be created along with the MacOS package of FastSurfer inside.
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
