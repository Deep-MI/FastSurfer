Corpus Callosum Pipeline
========================
A deep learning-based pipeline for automated segmentation, analysis, and shape analysis of the corpus callosum in brain MRI scans.
Also segments the fornix, localizes the anterior and posterior commissure (AC and PC) and standardizes the orientation of the brain.

Overview
--------
This pipeline combines localization and segmentation deep learning models to:
1. Extract and align midsagittal slices
2. Detect AC (Anterior Commissure) and PC (Posterior Commissure) points
3. Segment the corpus callosum
4. Perform advanced morphometry for corpus callosum, including subdivision, thickness analysis, and various shape metrics
5. Generate visualizations and measurements

FastSurfer-CC identifies the midsagittal slab for corpus callosum analysis, segments the corpus callosum and fornix,
localizes AC and PC landmarks for head-pose standardization, and derives area, thickness, length, curvature, volume,
and subdivision measures for downstream statistical analysis.

The output files are described [here](../OUTPUT_FILES.md#corpus-callosum-module).
The structure of the JSON files describing corpus callosum measures is documented below.
Advanced options, like custom subdivision schemes and quality control are described in the [FastSurfer-CC documentation](../../scripts/fastsurfer_cc.rst).

References
----------
If you use FastSurfer-CC in your research, please cite:

*Pollak C, Diers K, Estrada S, Kuegler D, Reuter M. FastSurfer-CC: A robust, accurate, and comprehensive framework for corpus callosum morphometry. Imaging Neuroscience 2026. https://doi.org/10.1162/IMAG.a.1221*

JSON Output Structure
---------------------
The pipeline generates two main JSON files with detailed measurements and analysis results:

### `stats/callosum.CC.midslice.json` (Middle Slice Analysis)
This file contains measurements from the middle sagittal slice and includes:

#### **Shape Measurements (single values):**
- `total_area`: Total corpus callosum area (mm²)
- `total_perimeter`: Total perimeter length (mm)
- `circularity`: Dimensionless shape circularity, defined below
- `cc_index`: Dimensionless corpus callosum index (CCI), defined below
- `midline_length`: Length along the corpus callosum midline (mm)
- `curvature`: Average curve of the midline (degrees), measured by angle between its sub-segments
- `curvature_body`: Average curve of the center 65% of the midline (degrees), measured by angle between its sub-segments

##### Corpus callosum index (`cc_index`)

FastSurfer-CC automates the corpus callosum index introduced as a practical
linear marker of callosal atrophy by
[Figueira et al. (2007)](https://doi.org/10.1590/S0004-282X2007000600001).
After aligning the sagittal CC contour to the AC--PC coordinate system, it
calculates

```{math}
\mathrm{CCI} = \frac{T_A + T_M + T_P}{L_{AP}},
```

where:

- {math}`L_{AP}` is the distance between the outermost anterior and posterior
  intersections of the longest anterior--posterior axis with the contour;
- {math}`T_A` and {math}`T_P` are the anterior and posterior callosal widths measured
  between the paired intersections on that axis; and
- {math}`T_M` is the width between the contour intersections of the perpendicular
  line through the midpoint of the anterior--posterior axis.

The CCI is dimensionless and approximately scale-independent. A larger value
means that the combined anterior, middle, and posterior widths are larger
relative to the anterior--posterior length. It is an aggregate shape measure,
not a direct estimate of callosal area or volume. FastSurfer's contour-based,
automated construction follows the geometry of the original manual measure;
values should therefore only be compared across data processed with consistent
orientation, segmentation, and contour settings.

The construction expects exactly four contour intersections with the
anterior--posterior axis. If that geometry cannot be established, FastSurfer
logs an error and reports `cc_index` as 0; this is a failure sentinel rather
than a biologically meaningful zero.

##### Circularity (`circularity`)

FastSurfer-CC calculates circularity from the total sagittal area {math}`A` and
contour perimeter {math}`P`:

```{math}
\mathrm{circularity} = \frac{4\pi A}{P^2}.
```

This dimensionless, scale-independent measure equals 1 for a perfect circle
and approaches 0 as a shape becomes increasingly elongated or irregular.
Because it combines area and perimeter, a lower value may reflect reduced area,
increased boundary length, or both; it does not localize the underlying shape
change. Perimeter is also sensitive to segmentation irregularity and contour
smoothing, so those settings should remain consistent in group or longitudinal
analyses.

Callosal circularity has been investigated as a shape marker in Alzheimer's
disease by [Ardekani et al. (2014)](https://doi.org/10.1007/s00429-013-0503-0)
and [Van Schependom et al. (2018)](https://doi.org/10.1016/j.nicl.2018.05.018).

#### **Subdivisions**
- `areas`: Areas of CC using an improved Hofer-Frahm sub-division method (mm²). This gives more consistent sub-segments while preserving the original ratios.
- `curvature_subsegments`: Average curve in the CC subsegments (see 'curvature')

#### **Thickness Analysis:**
- `thickness`: Average corpus callosum thickness (mm)
- `thickness_profile`: Thickness profile (mm) of the corpus callosum slice (100 thickness values by default, listed from anterior to posterior CC ends)

#### **Volume Measurements:**
- `cc_num_voxel`: Segmentation-based (masks) CC voxel count within a 5mm slab around the midsagittal plane (partial voxels at the edges are weighted to achieve exactly 5mm width). Multiply by `voxel_volume` to get the volume in mm³.
- `cc_volume`: Surface-based (contour) CC volume estimate in mm³, computed from the CC contours across all valid slices assuming 5mm slab width. Only reliable when `cc_num_failed_slices` is 0. `null` if fewer than 2 contour slices processed successfully.

#### **Anatomical Landmarks:**
All anatomical landmarks are given image voxel coordinates (LIA orientation)
- `ac_center`: Anterior commissure coordinates in `orig.mgz` voxel space
- `pc_center`: Posterior commissure coordinates in `orig.mgz` voxel space
- `ac_center_oriented_volume`: AC coordinates in standardized space (orient_volume.lta)
- `pc_center_oriented_volume`: PC coordinates in standardized space (orient_volume.lta)
- `ac_center_upright`: AC coordinates in upright space (cc_up.lta)
- `pc_center_upright`: PC coordinates in upright space (cc_up.lta)
- `landmark_source`: AC/PC landmark provenance: `"model"` (default) or `"supplied"` when `--ac_coords` and
  `--pc_coords` are passed
- `segmentation_source`: Upright CC segmentation provenance: `"model"` (default) or `"manual"` when
  `--segmentation_manedit` is passed

### `stats/callosum.CC.all_slices.json` (Multi-Slice Analysis)
This file contains comprehensive per-slice analysis when using `--slice_selection all`:

#### **Global Parameters:**
- `slices_in_segmentation`: Total number of slices in the segmentation volume
- `voxel_size`: Voxel dimensions [x, y, z] in mm
- `voxel_volume`: Volume of a single voxel in mm³
- `cc_num_failed_slices`: Number of slices for which surface processing failed
- `subdivision_method`: Method used for anatomical subdivision
- `num_thickness_points`: Number of points used for thickness estimation
- `subdivision_ratios`: Subdivision fractions used for regional analysis
- `contour_smoothing`: Gaussian sigma used for contour smoothing
- `slice_selection`: Slice selection mode used

#### **Per-Slice Data (`slices` array):**
Each slice entry contains the shape measurements, thickness analysis and sub-divisions as described above.
