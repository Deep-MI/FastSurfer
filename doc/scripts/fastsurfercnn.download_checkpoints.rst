FastSurferCNN: download_checkpoints.py
======================================

`download_checkpoints` is a script is a script to download the checkpoint files for all the various neural networks in FastSurfer. It can be used to install them during a native install. Otherwise checkpoints will be loaded during the first run of FastSurfer automatically.

The script also checks the download's md5 sums against pre-computed md5 sums. If files do not match against the md5 sum, they are downloaded again like a broken transfer. Checkpoints that already exist are not downloaded again, but checked: if their md5 sum does not match, the script logs a warning and keeps them.

Full commandline interface of FastSurferCNN/download_checkpoints.py
-------------------------------------------------------------------
.. argparse::
    :module: FastSurferCNN.download_checkpoints
    :func: make_parser
    :prog: FastSurferCNN/download_checkpoints.py
