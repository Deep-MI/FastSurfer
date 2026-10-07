tools: compare_subjects.py
==========================

``compare_subjects.py`` compares two FastSurfer subject directories by content, for example two runs of the same
input, or the results before and after an update of FastSurfer, FreeSurfer or the hardware. ``diff`` and checksums
do not help here, because the files record timestamps, the command line and versions in their headers. The script
compares what the files contain instead (voxels, surfaces, per-vertex data, transforms, statistics and annotations)
and reports how large each difference is.

The script is in the ``tools`` folder of a FastSurfer checkout and of the Docker and Singularity images
(``/fastsurfer/tools``); the macOS package does not include it. It needs FastSurfer's Python environment.

Usage:

.. code-block:: text

    python3 <fastsurfer_home>/tools/compare_subjects.py <subject_dir_1> <subject_dir_2>

It prints one line per file that differs or exists on one side only, and a summary. The exit code is 0 if everything
compared is identical, 1 if there are differences, and 2 if it could not compare at all.

Full commandline interface of tools/compare_subjects.py
-------------------------------------------------------
.. argparse::
    :module: tools.compare_subjects
    :func: make_parser
    :prog: tools/compare_subjects.py
