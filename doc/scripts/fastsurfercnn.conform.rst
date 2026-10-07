FastSurferCNN: conform.py
=========================

``conform.py`` conforms an MRI image the way FastSurfer does before the segmentation: to 8-bit intensities (uchar),
LIA orientation and isotropic voxels, on a cube large enough for the field of view. ``run_fastsurfer.sh`` does this
itself, so you only need the script to prepare or inspect images on their own, for example to check which images of
a dataset FastSurfer will resample, similar to FreeSurfer's ``mri_convert -c``.

With ``--check_only``, it only reports whether an image is already conformed and writes nothing. Without options, it
conforms to 1 mm voxels; ``run_fastsurfer.sh`` uses ``--vox_size min`` by default, so pass ``--vox_size min`` to get
the same result for high-resolution images.

Usage:

.. code-block:: text

    python3 <fastsurfer_home>/FastSurferCNN/data_loader/conform.py \
        -i <t1_path> -o <output_path> --vox_size min
    python3 <fastsurfer_home>/FastSurferCNN/data_loader/conform.py \
        -i <t1_path> --check_only --vox_size min

Full commandline interface of FastSurferCNN/data_loader/conform.py
------------------------------------------------------------------
.. argparse::
    :module: FastSurferCNN.data_loader.conform
    :func: make_parser
    :prog: FastSurferCNN/data_loader/conform.py
