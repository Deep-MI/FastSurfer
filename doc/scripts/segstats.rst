FastSurferCNN: segstats.py
==========================

`segstats.py` is a script that is equivalent to FreeSurfer's `mri_segstats`. However, it is faster and (automatically) scales very well to multi-processing scenarios.


Full commandline interface of FastSurferCNN/segstats.py
-------------------------------------------------------
.. argparse::
    :module: FastSurferCNN.segstats
    :func: make_arguments
    :prog: FastSurferCNN/segstats.py
