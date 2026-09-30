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
`run_fastsurfer.sh --sd <subjects_dir> --sid <subject_id> --t1 <t1_file>`. Shell commands in examples use a `bash`
fence, not `sh`, `shell` or no language. Code in other languages (`python`, `toml`, `json`, `yaml`, ...) keeps its
language.

## Placeholders

Placeholders are `<snake_case>`: lowercase words joined by underscores, in angle brackets, for example `<subject_id>`,
`<subjects_dir>`, `<t1_file>`, `<template_id>` or `<tpid_1>`. Use them in explanations and in prose, and use the same
name for the same thing:

| Placeholder                                | Meaning                                                                         |
|--------------------------------------------|---------------------------------------------------------------------------------|
| `<data_dir>`                               | folder with the input images                                                    |
| `<subjects_dir>`                           | output folder, passed to `--sd`                                                 |
| `<subject_id>`                             | subject name, passed to `--sid`                                                 |
| `<subject_dir>`                            | output folder of one subject, `<subjects_dir>/<subject_id>`                     |
| `<t1_file>`, `<t2_file>`                   | input images, passed to `--t1` and `--t2`                                       |
| `<license_file>`                           | FreeSurfer license file, passed to `--fs_license`                               |
| `<lesion_mask_file>`                       | lesion mask, passed to `--lesion_mask`                                          |
| `<fastsurfer_home>`, `<freesurfer_home>`   | FastSurfer checkout and FreeSurfer installation, where an explanation sets `FASTSURFER_HOME` or `FREESURFER_HOME` |
| `<fastsurfer_flags>`, `<docker_flags>`, `<singularity_flags>` | further flags of FastSurfer, `docker run` or `singularity exec` |
| `<torch_device>`                           | run-time device, passed to `--device`: `auto`, `cpu`, `cuda`, `cuda:1`, `mps`   |
| `<version>`                                | FastSurfer version without `v`, image tags are `deepmi/fastsurfer:<device>-v<version>` |
| `<device>`                                 | device part of an image tag: `cu<cuda_version>`, `rocm`, `cpu`, ...             |
| `<sif_file>`                               | Singularity image file                                                          |
| `<image_tag>`                              | full image reference, for example `deepmi/fastsurfer:<device>-v<version>`       |
| `<host_folder>`, `<container_folder>`      | the two sides of a mount that does not use the same path                        |
| `<template_id>`, `<tpid_1>`, `<t1_file_1>`, `<t2_file_1>` | longitudinal template, time points and their images, passed to `--tid`, `--tpids`, `--t1s`, `--t2s` |

Scripts whose output does not follow the `<subjects_dir>/<subject_id>` layout keep the placeholders of their `--help`.
For example, `run_fastsurfer_bids.py` writes to `<output_dir>`, whose subject folders are named after BIDS entities
(`sub-<label>_ses-<label>`) instead of `--sid`, so `<subjects_dir>` would suggest the wrong structure.

Do not use:

- other spellings inside angle brackets: `<subject id>`, `<fastsurfer-flags>`, `<templateID>`, `<path/to/output/dir>`,
- paths that look real: `/path/to/license.txt`,
- `#` for digits: `v#.#.#`, `cu###`,
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
  [example values](#example-values), for example `export FASTSURFER_HOME=$HOME/FastSurfer`.
- Variables an example defines itself are lower snake_case (`data_dir`, `output_dir`, `subject_id`); environment
  variables keep their upper-case names.
- In prose, environment variables are fine, and so are variables set by the fence the text describes. Otherwise, use a
  placeholder.

## Example values

Names are snake_case, except `subjectX` and the file names of container images.

| What                                    | Example value                                                                                                                  |
|-----------------------------------------|--------------------------------------------------------------------------------------------------------------------------------|
| Input data folder                       | `$HOME/my_mri_data`                                                                                                            |
| Output (subjects) folder, `--sd`        | `$HOME/my_fastsurfer_analysis`                                                                                                 |
| FreeSurfer license file, `--fs_license` | `$HOME/my_fs_license.txt`                                                                                                      |
| Subject, `--sid`                        | `subjectX`                                                                                                                     |
| T1 image, `--t1`                        | `$HOME/my_mri_data/subjectX/t1_weighted.nii.gz`                                                                                |
| T2 image, `--t2`                        | `$HOME/my_mri_data/subjectX/t2_weighted.nii.gz`                                                                                |
| Longitudinal, `--tid`, `--tpids`, `--t1s` | template `subjectX`, time points named by scan date (`YYYYMMDD`): `subjectX_20210315`, `subjectX_20230920`, images `$HOME/my_mri_data/subjectX/t1_weighted_20210315.nii.gz`, ... |
| Subject list                            | `$HOME/my_mri_data/subjects_list.txt`                                                                                          |
| Lesion mask, `--lesion_mask`            | `$HOME/my_mri_data/subjectX/lesion_mask.nii.gz`                                                                                |
| BIDS dataset                            | `$HOME/my_bids_dataset`                                                                                                        |
| `FASTSURFER_HOME` (FastSurfer checkout) | `$HOME/FastSurfer`                                                                                                             |
| `FREESURFER_HOME`                       | `/opt/freesurfer`                                                                                                              |
| Singularity image                       | `$HOME/my_singularity_images/fastsurfer-{{ FASTSURFER_VERSION }}.sif`, for the CPU image `fastsurfer-cpu-{{ FASTSURFER_VERSION }}.sif` |
| FastSurfer scripts (native)             | `$FASTSURFER_HOME/run_fastsurfer.sh`, likewise `brun_fastsurfer.sh`, `long_fastsurfer.sh`, ...                                 |

## Containers

- Mount host folders and files at the same path in the container, so the paths passed to FastSurfer are the host
  paths:
  - Docker: `-v $HOME/my_mri_data:$HOME/my_mri_data`,
  - Singularity: `-B $HOME/my_mri_data`,
  - the license file itself: `-v $HOME/my_fs_license.txt:$HOME/my_fs_license.txt`.
- Windows (PowerShell) cannot use the same path, so map `C:/Users/user/<folder>` to `/home/user/<folder>`, for example
  `-v C:/Users/user/my_mri_data:/home/user/my_mri_data`.

## Line length

Lines in code blocks are at most 80 characters, so the rendered boxes do not scroll. Count the rendered line: without
the indentation of the block itself (in lists and reST), and with `{{ FASTSURFER_VERSION }}` as the version (up to 10
characters, for example `2.6.0-dev0`). Break longer lines:

- shell commands, in examples and explanations: at a space, ending the line with ` \` and indenting the continuation,
- other code: where the language allows a line break, for example inside brackets,
- output and logs: at a space, indenting the continuation by two spaces.

Keep a line that cannot be broken without changing its meaning, for example one line of a subject list, a single URL
or a single path.

Help texts of scripts shown with `command-output` follow the same limit: wrap them at 80 characters in the script.

## Versions

Official Docker images only exist for releases, so where a command names a version (in image tags and file names), it
uses `{{ FASTSURFER_VERSION }}`. In the documentation of a release, this is that release; in the development
documentation, the latest release. Likewise, the NVIDIA image of that release is
`{{ CUDA_STRING }}-v{{ FASTSURFER_VERSION }}`: `{{ CUDA_STRING }}` is `DEFAULTS.CUDA` in `tools/Docker/build.py` of
that release (for example `cu128`), its default CUDA version (the image `latest` points to while that release is the
newest). `{{ CUDA_VERSION }}` is `DEFAULTS.CUDA_VERSION` of the same file, the CUDA version of that image (for example
`12.8`). Name another CUDA version (`cu118`) only where the text is about choosing one.
`fix_links` also renders a note on top of each code block that uses them (`fix_links_substitution_banners` in
`doc/conf.py`), with a separate text for `{{ FASTSURFER_VERSION }}`, `{{ CUDA_STRING }}` and both: that the commands
use the latest release (in the development documentation), and that images for other CUDA versions are available on
Docker Hub. Do not add such notes by hand.

reST files are not substituted, so commands there use the `latest` image instead: `deepmi/fastsurfer:latest` and
`$HOME/my_singularity_images/fastsurfer-latest.sif`.
