FastSurferCNN: segstats.py
==========================

`segstats.py` is a script that is equivalent to FreeSurfer's `mri_segstats`. However, it is faster and (automatically) scales very well to multi-processing scenarios.


Full commandline interface of FastSurferCNN/segstats.py
-------------------------------------------------------
.. argparse::
    :module: FastSurferCNN.segstats
    :func: make_arguments
    :prog: FastSurferCNN/segstats.py

FreeSurfer-compatible interfaces: mri_segstats.py and mri_brainvol_stats.py
---------------------------------------------------------------------------
For scripts written for FreeSurfer, ``FastSurferCNN/mri_segstats.py`` and ``FastSurferCNN/mri_brainvol_stats.py``
accept the options of FreeSurfer's ``mri_segstats`` and ``mri_brainvol_stats`` and run ``segstats.py`` with the
equivalent options. Options that have no equivalent in ``segstats.py`` are not listed; ``--print`` shows the
equivalent ``segstats.py`` call.

.. argparse::
    :module: FastSurferCNN.mri_segstats
    :func: make_arguments
    :prog: FastSurferCNN/mri_segstats.py

.. argparse::
    :module: FastSurferCNN.mri_brainvol_stats
    :func: make_arguments
    :prog: FastSurferCNN/mri_brainvol_stats.py
