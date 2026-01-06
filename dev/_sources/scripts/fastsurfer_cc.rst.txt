CorpusCallosum: fastsurfer_cc.py
================================

.. note::
   We recommend to run FastSurfer-CC with the standard `run_fastsurfer.sh` interfaces </overview/FLAGS.md>!


..
   [Note] To tell sphinx where in the documentation CorpusCallosum/README.md can be linked to, it needs to be included somewhere

.. include:: ../../CorpusCallosum/README.md
   :parser: fix_links.parser
   :start-line: 1

.. argparse::
   :module: CorpusCallosum.fastsurfer_cc
   :func: make_parser
   :prog: fastsurfer_cc.py

.. include:: ../overview/modules/CC.md
   :parser: fix_links.parser
