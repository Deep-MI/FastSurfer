CorpusCallosum: contour.py
==========================

This module provides the ``CCContour`` class for reading, writing, and
manipulating 2D corpus callosum contours together with per-vertex thickness
values. Typical template outputs (from ``fastsurfer_cc.py --save_template``)
emit one set per slice:

- ``contour_<idx>.txt``: CSV with header ``New contour, anterior_endpoint_idx=<a>, posterior_endpoint_idx=<p>`` followed by ``x,y`` rows.
- ``thickness_values_<idx>.txt``: CSV with header ``thickness`` and one value per contour vertex.
- ``thickness_measurement_points_<idx>.txt``: CSV with header ``vertex_idx`` listing the vertices where thickness was measured.

Key usage patterns
------------------

.. code-block:: python

   from CorpusCallosum.shape.contour import CCContour

   contour = CCContour(contour_points, thickness_values,
                       endpoint_idxs=(anterior_idx, posterior_idx),
                       resolution=1.0)
   contour.fill_thickness_values()   # interpolate missing values
   contour.smooth_contour(window_size=5)
   contour.save_contour("contour_0.txt")
   contour.save_thickness_values("thickness_values_0.txt")
   contour.save_thickness_measurement_points("thickness_measurement_points_0.txt")

Reference
---------

.. automodule:: CorpusCallosum.shape.contour
   :members: CCContour
   :undoc-members:
   :show-inheritance:

