Modules
=======

``run_fastsurfer.sh`` runs FastSurfer's modules one after the other: first the segmentation modules, which take a few
minutes on a GPU, then the surface reconstruction, which takes about an hour. All modules run by default; each page
describes what a module computes, what it needs, how to switch it off or adjust it, and how to cite it.

Segmentation modules:

- :doc:`Whole-brain segmentation (FastSurferVINN) <ASEGDKT>`, the core module the others build on
- :doc:`Corpus callosum (FastSurfer-CC) <CC>`
- :doc:`Cerebellum sub-segmentation (CerebNet) <CEREBNET>`
- :doc:`Hypothalamus sub-segmentation (HypVINN) <HYPVINN>`

Surface module:

- :doc:`Surface reconstruction (recon-surf) <SURFACE>`

Extension:

- :doc:`Lesion inpainting (LIT) <LIT>`, for images with large lesions

.. toctree::
    :hidden:

    ASEGDKT
    CC
    CEREBNET
    HYPVINN
    SURFACE
    LIT
