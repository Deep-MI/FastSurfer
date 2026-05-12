CorpusCallosum: fastsurfer_cc.py
================================
.. note::
   We recommend running FastSurfer-CC with the standard ``run_fastsurfer.sh`` interface (see :doc:`RUN_FASTSURFER`)!

   This page documents expert usage of FastSurfer-CC, which can be run independently with the advanced interface provided here. However, the FastSurfer segmentation is still required as input.


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
