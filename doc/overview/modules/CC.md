Corpus Callosum Pipeline
========================
A deep learning-based pipeline for automated segmentation, analysis, and shape analysis of the corpus callosum in brain MRI scans.
Also segments the fornix, localizes the anterior and posterior commissure (AC and PC) and standardizes the orientation of the brain.

Overview
--------
This pipeline combines localization and segmentation deep learning models to:
1. Detect AC (Anterior Commissure) and PC (Posterior Commissure) points
2. Extract and align midplane slices
3. Segment the corpus callosum
4. Perform advanced morphometry for corpus callosum, including subdivision, thickness analysis, and various shape metrics
5. Generate visualizations and measurements

The output files are described [here](../OUTPUT_FILES.md#corpus-callosum-module).
The structure of the JSON files describing corpus callosum measures is documented below.
Advanced options, like custom subdivision schemes and quality control are described in the [FastSurfer-CC documentation](../../scripts/fastsurfer_cc.rst).

JSON Output Structure
---------------------
The pipeline generates two main JSON files with detailed measurements and analysis results:

### `stats/callosum.CC.midslice.json` (Middle Slice Analysis)
This file contains measurements from the middle sagittal slice and includes:

#### **Shape Measurements (single values):**
- `total_area`: Total corpus callosum area (mm²)
- `total_perimeter`: Total perimeter length (mm)
- `circularity`: Shape circularity measure (4π × area / perimeter²)
- `cc_index`: Corpus callosum shape index (length/width ratio)
- `midline_length`: Length along the corpus callosum midline (mm)
- `curvature`: Average curve of the midline (degrees), measured by angle between its sub-segments
- `curvature_body`: Average curve of the center 65% of the midline (degrees), measured by angle between its sub-segments

#### **Subdivisions**
- `areas`: Areas of CC using an improved Hofer-Frahm sub-division method (mm²). This gives more consistent sub-segments while preserving the original ratios.
- `curvature_subsegments`: Average curve in the CC subsegments (see 'curvature')

#### **Thickness Analysis:**
- `thickness`: Average corpus callosum thickness (mm)
- `thickness_profile`: Thickness profile (mm) of the corpus callosum slice (100 thickness values by default, listed from anterior to posterior CC ends)

#### **Volume Measurements (when multiple slices processed):**
- `cc_5mm_volume`: Total CC volume within 5mm slab using voxel counting (mm³)
- `cc_5mm_volume_pv_corrected`: Volume with partial volume correction using CC contours (mm³)

#### **Anatomical Landmarks:**
All anatomical landmarks are given image voxel coordinates (LIA orientation)
- `ac_center`: Anterior commissure coordinates in original image space (orig.mgz)
- `pc_center`: Posterior commissure coordinates in original image space (orig.mgz)
- `ac_center_oriented_volume`: AC coordinates in standardized space (orient_volume.lta)
- `pc_center_oriented_volume`: PC coordinates in standardized space (orient_volume.lta)
- `ac_center_upright`: AC coordinates in upright space (cc_up.lta)
- `pc_center_upright`: PC coordinates in upright space (cc_up.lta)

### `stats/callosum.CC.all_slices.json` (Multi-Slice Analysis)
This file contains comprehensive per-slice analysis when using `--slice_selection all`:

#### **Global Parameters:**
- `slices_in_segmentation`: Total number of slices in the segmentation volume
- `voxel_size`: Voxel dimensions [x, y, z] in mm
- `subdivision_method`: Method used for anatomical subdivision
- `num_thickness_points`: Number of points used for thickness estimation
- `subdivision_ratios`: Subdivision fractions used for regional analysis
- `contour_smoothing`: Gaussian sigma used for contour smoothing
- `slice_selection`: Slice selection mode used

#### **Per-Slice Data (`slices` array):**
Each slice entry contains the shape measurements, thickness analysis and sub-divisions as described above.
