# Documentation conventions

These conventions apply to the documentation sources: the Markdown and reST files in `doc/` and the files they
include with `.. include::` (for example `README.md`, `tools/Docker/README.md` and `recon_surf/README.md`). This file
itself is not part of the built documentation (`exclude_patterns` in `doc/conf.py`).

## Examples and explanations

Code blocks are either examples or explanations, and the two must look different.

- **Examples** are commands a reader can copy and run with at most small changes. They use a `bash` fence and the
  [example values](#example-values). Introduce them with "For example:". Sphinx adds a copy button. A value that no
  example can supply, such as the reader's GitHub user name (`<username>`), stays a placeholder.
- **Explanations** show the general form of a command. They use a `text` fence and [placeholders](#placeholders).
  Introduce them with "Usage:". Sphinx adds no copy button (`copybutton_selector` in `doc/conf.py`), and a `text`
  fence is not highlighted, so the two kinds also differ on GitHub.
- **Output** of commands, logs, file trees and file contents use a `text` fence as well, and commands in them have no
  prompt (`$`).

Explanations name scripts without a path, for example
`run_fastsurfer.sh --sd <subjects_dir> --sid <subject_id> --t1 <t1_path>`. Shell commands in examples use a
`bash` fence, not `sh`, `shell` or no language. Code in other languages (`python`, `toml`, `json`, `yaml`, ...) keeps
its language.

## Placeholders

Placeholders are `<snake_case>`: lowercase words joined by underscores, in angle brackets, for example `<subject_id>`,
`<subjects_dir>`, `<t1_path>`, `<template_id>` or `<tpid_1>`. Use them in explanations and in prose, and use the
same name for the same thing. A placeholder for a file or a folder stands for its full path (absolute or relative to
the current directory), and its suffix says which of the two it is:

- `_path` for a file: `<t1_path>`, `<freesurfer_license_path>`, `<sif_path>`,
- `_dir` for a folder: `<subjects_dir>`, `<subject_dir>`, `<data_dir>`,
- `_name` for a single file or folder name that is not a path: `<some_dir_name>`.

`<fastsurfer_home>` and `<freesurfer_home>` are named after the environment variables they stand for and keep that
name.

| Placeholder                                               | Meaning                                                                                              |
|-----------------------------------------------------------|------------------------------------------------------------------------------------------------------|
| `<data_dir>`                                              | folder with the input images                                                                         |
| `<subjects_dir>`                                          | output folder, passed to `--sd`                                                                      |
| `<subject_id>`                                            | subject name, passed to `--sid`                                                                      |
| `<subject_dir>`                                           | output folder of one subject, `<subjects_dir>/<subject_id>`                                          |
| `<t1_path>`, `<t2_path>`                                  | input images, passed to `--t1` and `--t2`                                                            |
| `<freesurfer_license_path>`                               | FreeSurfer license file, passed to `--fs_license`                                                    |
| `<lesion_mask_path>`                                      | lesion mask, passed to `--lesion_mask`                                                               |
| `<subjects_list_path>`                                    | subjects list file, passed to `--subjects_list`                                                      |
| `<fastsurfer_home>`, `<freesurfer_home>`                  | FastSurfer checkout and FreeSurfer installation, where an explanation sets `FASTSURFER_HOME` or `FREESURFER_HOME` |
| `<fastsurfer_flags>`, `<docker_flags>`, `<singularity_flags>` | further flags of FastSurfer, `docker run` or `singularity exec`                                  |
| `<torch_device>`                                          | run-time device, passed to `--device`: `auto`, `cpu`, `cuda`, `cuda:1`, `mps`                        |
| `<version>`                                               | FastSurfer version without `v`, image tags are `deepmi/fastsurfer:<device>-v<version>`              |
| `<device>`                                                | device part of an image tag: `cu<cuda_version>`, `rocm`, `cpu`, ...                                  |
| `<sif_path>`                                              | Singularity image file                                                                               |
| `<image_tag>`                                             | full image reference, for example `deepmi/fastsurfer:<device>-v<version>`                            |
| `<host_dir>`, `<container_dir>`                           | the two sides of a mount that does not use the same path                                             |
| `<template_id>`, `<tpid_1>`, `<t1_path_1>`, `<t2_path_1>` | longitudinal template, time points and their images, passed to `--tid`, `--tpids`, `--t1s`, `--t2s` |

Scripts whose output does not follow the `<subjects_dir>/<subject_id>` layout keep the placeholders of their
`--help`. For example, `bids_fastsurfer.py` writes to `<output_dir>`, whose subject folders are named after
BIDS entities (`sub-<label>_ses-<label>`) instead of `--sid`, so `<subjects_dir>` would suggest the wrong
structure. Its positional arguments keep the names of the BIDS-App specification (`bids_dir output_dir`), and their
placeholders `<bids_dir>` and `<output_dir>` follow the rules above.

Do not use:

- other spellings inside angle brackets: `<subject id>`, `<fastsurfer-flags>`, `<templateID>`, `<path/to/output/dir>`,
- made-up paths, except the defaults `/path/to/FastSurfer`, `/path/to/freesurfer` and
  `/path/to/your/freesurfer/license_file`, see [Shell variables](#shell-variables),
- Version numbers should not be in the documentation, instead use {{ FASTSURFER_VERSION }} or {{ CUDA_STRING }} in
  markdown files, or the other substitutions in [Versions](#versions).
- `{name}`, except in `python` fences, where it is Python format syntax.

`{{ NAME }}` is only valid where it is actually substituted, so `NAME` must be a key of `myst_substitutions` in
`doc/conf.py`. MyST substitutes it in prose, and the `fix_links` extension in fenced and inline code (there, only
string values). Neither substitutes in link targets or raw HTML. The Sphinx documentation is what counts, GitHub shows
`{{ NAME }}` unchanged.

## Shell variables

- In a `bash` fence, use a variable only if the same fence sets it (`name=...`, `export name=...`, `for name in ...`,
  `read name`). Variables the shell provides (`$HOME`, `$PWD`, `$USER`, `$PATH`) are always fine.
- The setup variables `FASTSURFER_HOME`, `FREESURFER_HOME` and `SUBJECTS_DIR` are set in the fence that uses them, or
  a comment in that fence says that they must be set. Examples set them at the top of the fence to the
  [example values](#example-values). `FASTSURFER_HOME` and `FREESURFER_HOME` keep a value the reader already set, and
  otherwise default to a made-up path that shows it has to be replaced:
  `export FASTSURFER_HOME=${FASTSURFER_HOME:-/path/to/FastSurfer}` and
  `export FREESURFER_HOME=${FREESURFER_HOME:-/path/to/freesurfer}`.
- Examples that clone FastSurfer set `FASTSURFER_HOME` first and clone into it:
  `git clone --branch stable https://github.com/Deep-MI/FastSurfer.git $FASTSURFER_HOME`.
- The location of the FreeSurfer license file cannot be guessed either. Examples that pass it set
  `freesurfer_license=${freesurfer_license:-/path/to/your/freesurfer/license_file}` at the top of the fence and use
  `$freesurfer_license`: `--fs_license $freesurfer_license`. Examples with `--seg_only` do not need the license, unless
  they add `--tal_reg` or a `--t2` image (both run FreeSurfer registrations), so they neither set, mount nor pass it.
- Variables an example defines itself are lower snake_case (`data_dir`, `output_dir`, `subject_id`,
  `freesurfer_license`); environment variables keep their upper-case names.
- In prose, environment variables are fine, and so are variables set by the fence the text describes. Otherwise, use a
  placeholder.

## Example values

Names are snake_case, except `subjectX` and the file names of container images.

| What                                      | Example value                                                                                                                  |
|-------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------|
| Input data folder                         | `$HOME/my_mri_data`                                                                                                            |
| Output (subjects) folder, `--sd`          | `$HOME/my_fastsurfer_analysis`                                                                                                 |
| FreeSurfer license file, `--fs_license`   | `$freesurfer_license`, by default `/path/to/your/freesurfer/license_file`                                                      |
| Subject, `--sid`                          | `subjectX`                                                                                                                     |
| T1 image, `--t1`                          | `$HOME/my_mri_data/subjectX/t1_weighted.nii.gz`                                                                                |
| T2 image, `--t2`                          | `$HOME/my_mri_data/subjectX/t2_weighted.nii.gz`                                                                                |
| Longitudinal, `--tid`, `--tpids`, `--t1s` | template `subjectX`, time points named by scan date (`YYYYMMDD`): `subjectX_20210315`, `subjectX_20230920`, images `$HOME/my_mri_data/subjectX/t1_weighted_20210315.nii.gz`, ... |
| Subjects list, `--subjects_list`          | `$HOME/my_mri_data/subjects_list.txt`                                                                                          |
| Lesion mask, `--lesion_mask`              | `$HOME/my_mri_data/subjectX/lesion_mask.nii.gz`                                                                                |
| BIDS dataset                              | `$HOME/my_bids_dataset`                                                                                                        |
| `FASTSURFER_HOME` (FastSurfer checkout)   | `${FASTSURFER_HOME:-/path/to/FastSurfer}`                                                                                      |
| `FREESURFER_HOME`                         | `${FREESURFER_HOME:-/path/to/freesurfer}`                                                                                      |
| Singularity image                         | `$HOME/my_singularity_images/fastsurfer-{{ CUDA_STRING }}-v{{ FASTSURFER_VERSION }}.sif`, for the CPU image `fastsurfer-cpu-v{{ FASTSURFER_VERSION }}.sif` |
| FastSurfer scripts (native)               | `$FASTSURFER_HOME/run_fastsurfer.sh`, likewise `brun_fastsurfer.sh`, `long_fastsurfer.sh`, ...                                 |

## Containers

- Mount host folders and files at the same path in the container, so the paths passed to FastSurfer are the host
  paths:
  - Docker: `-v $HOME/my_mri_data:$HOME/my_mri_data`,
  - Singularity: `-B $HOME/my_mri_data`,
  - the license file itself: `-v $freesurfer_license:$freesurfer_license` or `-B $freesurfer_license`.
- Windows (PowerShell) cannot use the same path, so map `C:/Users/user/<dir_name>` to `/home/user/<dir_name>`, for
  example `-v C:/Users/user/my_mri_data:/home/user/my_mri_data`. The license file is mounted from its made-up path:
  `-v C:/path/to/your/freesurfer/license_file:/home/user/freesurfer_license`.

## Line length

Lines in code fences in the documentation are at most 80 characters, so there is no scroll bar. Count the rendered
line, without the indentation of the block itself (in lists and reST). Break longer lines:

- shell commands, in examples and explanations: at a space, ending the line with ` \` and indenting the continuation,
- other code: where the language allows a line break, for example inside brackets,
- output and logs: at a space, indenting the continuation by two spaces.

Keep a line that cannot be broken without changing its meaning, for example one line of a subjects list, a single URL
or a single path.

Help texts of scripts shown with `command-output` follow the same limit: wrap them at 80 characters in the script.

## Versions

Official Docker images only exist for releases, so where a command names a version (in image tags and file names), it
uses `{{ FASTSURFER_VERSION }}`. In the documentation of a release, this is that release; in the development
documentation, the latest release. Likewise, the NVIDIA image of that release is
`{{ CUDA_STRING }}-v{{ FASTSURFER_VERSION }}`: `{{ CUDA_VERSION }}` is `tool.cuda.version` in `pyproject.toml` of that
release, the CUDA version of the image `latest` points to while that release is the newest, and
`{{ CUDA_STRING }}` the PyTorch backend of that version (`cu` and the version without dots). Name another CUDA
version only where the text is about choosing one; `tools/Docker/build.py --print_supported cuda|rocm` lists the
supported versions.
`fix_links` also renders a note on top of each code block that uses them (`fix_links_substitution_banners` in
`doc/conf.py`), with a separate text for `{{ FASTSURFER_VERSION }}`, `{{ CUDA_STRING }}` and both: that the commands
use the latest release (in the development documentation), and that images for other CUDA versions are available on
Docker Hub. Do not add such notes by hand. Where the version is not about an image, such as the folder of the macOS
package, use `{{ PACKAGE_VERSION }}`, the same version without the note.

What this tree builds comes from this tree, not from that release: `{{ CUDA_DEFAULT_STRING }}` and
`{{ ROCM_DEFAULT_STRING }}` (with `{{ CUDA_DEFAULT_VERSION }}` and `{{ ROCM_DEFAULT_VERSION }}`) are the default CUDA
and ROCm images of `pyproject.toml`, and `{{ CUDA_LEGACY_STRING }}` is the image for older GPUs, from
`FastSurferCNN/gpu_support.py`. Use them where the text is about building images or choosing one.

The software in the images of that release, which is also what the native installation clones (`--branch stable`),
comes with versions from the same tree: `{{ PYTHON_VERSION }}` is `tool.python.version` in `pyproject.toml`,
`{{ UBUNTU_VERSION }}` the Ubuntu version of `tool.docker.runtime_base` in `pyproject.toml`, and
`{{ FREESURFER_VERSION }}` is `tool.freesurfer.version` in `pyproject.toml`. Code blocks that use them get no note.

reST files are not substituted, so commands there use the `latest` image instead: `deepmi/fastsurfer:latest` and
`$HOME/my_singularity_images/fastsurfer-latest.sif`.
