CorpusCallosum: fastsurfer_cc.py
================================
.. note::
   We recommend running FastSurfer-CC with the standard ``run_fastsurfer.sh`` interface (see :doc:`RUN_FASTSURFER`)!

   This page documents expert usage of FastSurfer-CC, which can be run independently with the advanced interface
   provided here. By default, it requires ``mri/orig.mgz`` and ``mri/aseg.auto_noCCseg.mgz`` from a FastSurfer
   subject; use ``--conformed_name`` and ``--aseg_name`` to override those input paths.


..
   [Note] To tell sphinx where in the documentation CorpusCallosum/README.md can be linked to, it needs to be included somewhere

.. include:: ../../CorpusCallosum/README.md
   :parser: fix_links.parser
   :start-line: 2

Full command-line interface
---------------------------
The following section provides a detailed overview of the command-line interface for the FastSurfer-CC pipeline, including all available flags and options.

.. argparse::
   :module: CorpusCallosum.fastsurfer_cc
   :func: make_parser
   :prog: fastsurfer_cc.py


Midplane extraction
-------------------
The ``--midplane_method`` flag controls how the corpus callosum pipeline refines the midsagittal plane before segmentation. This is implemented in ``CorpusCallosum/registration/midsagittal_plane_alignment.py``.
When the corpus callosum segmentation does not align well with the mid-sagittal plane, this option can be changed for better results.

Available modes are:

- ``fsaverage_symmetry``: (default) Align to fsaverage, then search for a small left-right shift that minimizes mirrored aseg-label mismatch near the midline
- ``fsaverage``: Align to fsaverage without additional refinement (matches publication)
- ``center``: use the geometric center of the input volume without additional refinement
- ``fsaverage_distance_map``: fit a midsagittal plane from left/right distance-map symmetry in fsaverage space

The refinement after fsaverage alignment is intentionally conservative, expected to only make adjustments for unusual anatomies, or significant asymmetry.

Supplying AC/PC landmarks
~~~~~~~~~~~~~~~~~~~~~~~~~
The expert interface accepts paired ``--ac_coords X Y Z`` and ``--pc_coords X Y Z`` arguments. Values are
floating-point voxel coordinates in ``orig.mgz`` voxel space. When supplied, the landmark network is skipped and the
points are used for segmentation conditioning, morphometry, QC, orientation transforms, and output measurements.

The selected midsagittal plane is minimally rotated and translated so that it contains both 3D landmarks exactly. A
plane adjustment larger than 15 degrees emits a warning because this commonly indicates a coordinate-space error.

For example:

.. code-block:: bash

    python3 CorpusCallosum/fastsurfer_cc.py \
        --sd /data/subjects \
        --sid sub001 \
        --ac_coords 127.4 126.8 128.1 \
        --pc_coords 127.9 103.6 128.7 \
        --upright_volume mri/upright_volume.mgz

For Docker or Singularity/Apptainer, append the same two options to the corresponding expert command. Coordinates are
interpreted in the voxel space of the subject's input ``mri/orig.mgz``, not scanner RAS coordinates. Both points are
required, must be finite and distinct, and must lie inside the image. A left-right AC-PC line is rejected because it
does not define a stable sagittal plane. These options are available through the expert interface and are not exposed
by ``run_fastsurfer.sh``.

Manual CC edits
---------------
.. warning::
   Choose and record any supplied AC/PC coordinates before creating a manual CC correction. The automatic run and
   every edit rerun must use exactly the same ``--ac_coords``, ``--pc_coords``, and ``--midplane_method`` values. If
   these values change, regenerate ``mri/upright_volume.mgz`` and ``mri/callosum.CC.upright.mgz`` and recreate or
   explicitly rebase the manual correction on the new upright image. Do not apply an edit created on a different
   upright plane. Because upright files use a standardized affine, FastSurfer-CC cannot reliably detect every stale
   edit from file geometry alone.

To reprocess a manual CC correction, first copy the automatic upright segmentation:

.. code-block:: bash

    SUBJECT_DIR=/data/subjects/sub001
    cp "$SUBJECT_DIR/mri/callosum.CC.upright.mgz" \
       "$SUBJECT_DIR/mri/callosum.CC.upright.manedit.mgz"

Edit label 192 in ``mri/callosum.CC.upright.manedit.mgz`` using ``mri/upright_volume.mgz`` as the anatomical reference.
The manual file may also contain fornix label 250. Then rerun the expert command with the manual input:

.. code-block:: bash

    python3 CorpusCallosum/fastsurfer_cc.py \
        --sd /data/subjects \
        --sid sub001 \
        --segmentation_manedit mri/callosum.CC.upright.manedit.mgz \
        --upright_volume mri/upright_volume.mgz \
        --qc_image qc_snapshots/callosum.png \
        --thickness_image qc_snapshots/callosum.thickness.png

If the manual segmentation contains any voxels with fornix label 250, FastSurfer-CC treats its CC and fornix as
self-contained input. If label 250 is absent, FastSurfer-CC instead retains the fornix from the automatic upright and
original-space CC segmentations, which must therefore exist from a previous run. It recomputes all CC-derived results
and writes ``mri/callosum.CC.orig.manedit.mgz``. The top-level ``run_fastsurfer.sh --edits`` workflow also uses this
derived file for downstream inpainting. Any label-250 voxels make the manual file authoritative for the entire
fornix; remove all label-250 voxels from the manual file to use the automatic fornix instead. Do not manually edit the
original-space file.

.. note::
   A direct expert invocation produces the CC outputs, including ``callosum.CC.orig.manedit.mgz``, but does not paint
   them into the broader aseg outputs. Use the top-level ``run_fastsurfer.sh --edits`` workflow when that downstream
   integration is required. The top-level workflow does not currently accept supplied AC/PC coordinates.

The complete Docker edit rerun is:

.. code-block:: bash

    SUBJECTS_DIR=/data/fastsurfer
    SID=sub001

    docker run --gpus all --rm \
        --user "$(id -u):$(id -g)" \
        --volume "$SUBJECTS_DIR:/output" \
        --entrypoint /fastsurfer/tools/Docker/entrypoint.sh \
        deepmi/fastsurfer:latest \
        python3 /fastsurfer/CorpusCallosum/fastsurfer_cc.py \
        --sd /output \
        --sid "$SID" \
        --segmentation_manedit mri/callosum.CC.upright.manedit.mgz \
        --upright_volume mri/upright_volume.mgz \
        --qc_image qc_snapshots/callosum.png \
        --thickness_image qc_snapshots/callosum.thickness.png

The example above resolves the relative manual path inside ``/output/$SID``. A manual segmentation stored elsewhere
on the host can instead be mounted read-only and passed by its absolute container path:

.. code-block:: bash

    MANUAL_EDIT=/data/annotations/sub001_cc_manedit.mgz

    docker run --gpus all --rm \
        --user "$(id -u):$(id -g)" \
        --volume "$SUBJECTS_DIR:/output" \
        --volume "$MANUAL_EDIT:/manual-edit.mgz:ro" \
        --entrypoint /fastsurfer/tools/Docker/entrypoint.sh \
        deepmi/fastsurfer:latest \
        python3 /fastsurfer/CorpusCallosum/fastsurfer_cc.py \
        --sd /output \
        --sid "$SID" \
        --segmentation_manedit /manual-edit.mgz \
        --qc_image qc_snapshots/callosum.png

The complete Singularity edit rerun is:

.. code-block:: bash

    SUBJECTS_DIR=/data/fastsurfer
    SID=sub001
    FASTSURFER_SIF=/containers/fastsurfer-gpu.sif

    singularity exec --nv --no-mount home,cwd -e \
        --bind "$SUBJECTS_DIR:/output" \
        "$FASTSURFER_SIF" \
        python3 /fastsurfer/CorpusCallosum/fastsurfer_cc.py \
        --sd /output \
        --sid "$SID" \
        --segmentation_manedit mri/callosum.CC.upright.manedit.mgz \
        --upright_volume mri/upright_volume.mgz \
        --qc_image qc_snapshots/callosum.png \
        --thickness_image qc_snapshots/callosum.thickness.png

An automatic CC run is only required when the manual segmentation does not contain fornix label 250. If the
segmentation used as the editing reference was generated with supplied AC/PC landmarks, append the exact same
``--ac_coords X Y Z --pc_coords X Y Z`` arguments to the edit command.



Quality Control
---------------
The pipeline can produce a dedicated quality control image, showing the CC contour, AC/PC landmarks and thickness estimation.
For this use the ``--qc_image`` flag.
Additionally, the surface outputs, e.g. ``--thickness_image``, can be used to visualize the CC thickness and also inform quality control.
Finally, to confirm the alignment of the CC on the mid-sagittal plane, we can output the upright volume with ``--upright_volume`` flag.
In this image the mid-sagittal plane is at voxel coordinate 128 in the LR direction.

An example call with all quality control outputs is:
.. code-block:: bash

    python3 fastsurfer_cc.py --sd /data/subjects --sid sub001 \
        --qc_image /data/qc/sub001/qc_snapshots/callosum.png \
        --thickness_image /data/qc/sub001/qc_snapshots/callosum.thickness.png \
        --upright_volume /data/qc/sub001/mri/upright_volume.mgz

Custom Subdivision Schemes
--------------------------
The pipeline supports custom subdivision schemes for the corpus callosum with the ``--subdivisions`` flag.
The fractions are relative to the total length of the corpus callosum (midline length).
The default is to use the shape-based subdivision scheme (recommended) and the Hofer-Frahms convention.

We can, for example divide the CC into 4 equal parts with the shape-based subdivision scheme:

.. code-block:: bash

    python3 fastsurfer_cc.py --sd /data/subjects --sid sub001 \
        --subdivision_method shape \
        --subdivisions 0.25 0.5 0.75

Analysis Modes
--------------
The pipeline supports different analysis modes that determine the type of template data generated.

3D Analysis
~~~~~~~~~~~
When running the main pipeline with ``--slice_selection all`` and ``--save_template_dir``, a complete 3D template is generated:

.. code-block:: bash

    # Generate 3D template data
    python3 fastsurfer_cc.py --sd /data/subjects --sid sub001 \
        --slice_selection all \
        --save_template_dir /data/templates/sub001

The template files can be used to visualize the corpus callosum in 3D with the :doc:`cc_visualization` script.

**Benefits:**
- Enables volumetric thickness analysis
- Supports advanced 3D visualizations with proper surface topology
- Creates surface and overlay files viewable in freeview and for integration with other tools


2D Analysis
~~~~~~~~~~~
When using ``--slice_selection middle`` or a specific slice number with ``--save_template_dir``:

.. code-block:: bash

    # Generate 2D template data (middle slice)
    python3 fastsurfer_cc.py --sd /data/subjects --sid sub001 \
        --slice_selection middle \
        --save_template_dir /data/templates/sub001

**Benefits:**
- Faster processing for single-slice analysis
- 2D visualization is most suitable for displaying downstream statistics
- Compatibility with classical corpus callosum studies

Choosing Analysis Mode
~~~~~~~~~~~~~~~~~~~~~~
**Use 3D Analysis (``--slice_selection all``) when:**
- Surface-based visualization is required
- Comprehensive thickness mapping across the entire corpus callosum is desired
- Generating a 3D template, e.g. for mesh visualization or 3D thickness mapping

**Use 2D Analysis (``--slice_selection middle`` or specific slice) when:**
- Faster processing is preferred
- A specific slice is selected (e.g. to correct for errors in mid-sagittal plane selection)
- Generating a 2D template, e.g. for 2D thickness mapping or plotting of cross-sectional statistics

For advanced 3D visualization options, see the :doc:`cc_visualization` documentation.
