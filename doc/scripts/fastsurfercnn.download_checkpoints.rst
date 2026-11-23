FastSurferCNN: download_checkpoints.py
======================================

`download_checkpoints.py` is a script to download the checkpoint files for all the various neural networks in FastSurfer. It can be used to install them during a native install. Otherwise checkpoints will be loaded during the first run of FastSurfer automatically.

The script checks all downloads against the md5 sums as verified by the FastSurfer maintainers. If files do not match, they are downloaded again like a broken transfer. Checkpoints that already exist are checked as well: if their md5 sum does not match, the script replaces them with the published file.

Full commandline interface of FastSurferCNN/download_checkpoints.py
-------------------------------------------------------------------
.. argparse::
    :module: FastSurferCNN.download_checkpoints
    :func: make_parser
    :prog: FastSurferCNN/download_checkpoints.py
