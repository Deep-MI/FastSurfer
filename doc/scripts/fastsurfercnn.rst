FastSurferCNN: run_prediction.py
================================

.. note::
   We recommend to run the surface pipeline with the standard `run_fastsurfer.sh` interfaces!

.. include:: ../../FastSurferCNN/README.md
   :parser: fix_links.parser
   :relative-docs: .
   :relative-images:
   :start-after: <!-- after inference heading -->
   :end-before: <!-- before generate_hdf5 -->

Full commandline interface of FastSurferCNN/run_prediction.py
-------------------------------------------------------------
.. argparse::
    :module: FastSurferCNN.run_prediction
    :func: make_parser
    :prog: FastSurferCNN/run_prediction.py
