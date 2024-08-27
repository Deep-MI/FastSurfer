HypVINN: run_prediction.py
==========================

.. note::
   We recommend to run HypVINN with the standard `run_fastsurfer.sh` interfaces!

The `HypVINN/run_prediction.py` script enables the inference with HypVINN. In most
situations, it will be called from `run_fastsurfer.sh` a direct call to
`HypVINN/run_prediction.py` is not needed.

.. argparse::
    :module: HypVINN.run_prediction
    :func: option_parse
    :prog: HypVINN/run_prediction.py


.. include:: ../../HypVINN/README.md
    :parser: fix_links.parser
    :start-line: 1
