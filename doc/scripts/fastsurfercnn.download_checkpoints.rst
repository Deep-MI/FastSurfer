FastSurferCNN: download_checkpoints.py
======================================

`download_checkpoints` is a script is a script to download the checkpoint files for all the various neural networks in FastSurfer. It can be used to install them during a native install. Otherwise checkpoints will be loaded during the first run of FastSurfer automatically.

Full commandline interface of FastSurferCNN/download_checkpoints.py
-------------------------------------------------------------------
.. argparse::
    :module: FastSurferCNN.download_checkpoints
    :func: make_parser
    :prog: FastSurferCNN/download_checkpoints.py
