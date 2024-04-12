CerebNet: run_prediction.py
===========================

.. note::
   We recommend to run CerebNet with the standard `run_fastsurfer.sh` interfaces!

The `CerebNet/run_prediction.py` script enables the inference with CerebNet. In most
situations, it will be called from `run_fastsurfer.sh` a direct call to
`CerebNet/run_prediction.py` is not needed.

.. argparse::
    :module: CerebNet.run_prediction
    :func: setup_options
    :prog: CerebNet/run_prediction.py


.. include:: ../../CerebNet/README.md
    :parser: fix_links.parser
    :start-line: 1
