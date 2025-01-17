##########################
Introduction to FastSurfer
##########################

We are excited that you are here. In this documentation we will help you get started with FastSurfer!

FastSurfer is an open-source AI software tool to extract quantiative measurements from human brain MRI (T1-weighted) images.
You will learn about it's different segmentation and surface modules and how to install and run it natively or in the
recommended Docker or Singularity images. But first let us tell you why we think FastSurfer is great:

* FastSurfer uses dedicated and fast AI methods (developed in-house).
* It is thoroughly validated across different scanners, field-strenghts, T1 sequences, ages, diesease, ...
* FastSurfer is fully open-source using a permissive Apache license.
* It is compatible with FreeSurfer, enabling FreeSurfer downstream tools to work directly.
* It is much faster and provides increased reliability and sensitivity of the derived measures.
* It natively supports high-resolution images (down to around 0.7mm) at high accuracy.
* It has modules for full-brain (aseg+aparcDKT), cerebellum and hypothalamic sub-segmentations.
* The segmentation modules run within minutes and provide partial-volume corrected stats.
* It has an optimized surface stream for cortical thickness analysis and improved correspondence.

So, there is really no reason why you should not try this out!

.. include:: ../../README.md
    :parser: fix_links.parser
    :relative-docs: .
    :relative-images:
    :start-after: <!-- start of system requirements -->
