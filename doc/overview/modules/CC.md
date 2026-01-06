# Corpus Callosum Pipeline

A deep learning-based pipeline for automated segmentation, analysis, and shape analysis of the corpus callosum in brain MRI scans.
Also segments the fornix, localizes the anterior and posterior commissure (AC and PC) and standardizes the orientation of the brain.

## Overview

This pipeline combines localization and segmentation deep learning models to:
1. Detect AC (Anterior Commissure) and PC (Posterior Commissure) points
2. Extract and align midplane slices
3. Segment the corpus callosum
4. Perform advanced morphometry for corpus callosum, including subdivision, thickness analysis, and various shape metrics
5. Generate visualizations and measurements

## Analysis Modes

The pipeline supports different analysis modes that determine the type of template data generated.

### 3D Analysis

When running the main pipeline with `--slice_selection all` and `--save_template_dir`, a complete 3D template is generated:

```bash
# Generate 3D template data
python3 fastsurfer_cc.py --sd /data/subjects --sid sub001 \
    --slice_selection all \
    --save_template_dir /data/templates/sub001
```

This creates:
- `contour_<idx>.txt`: Multi-slice contour data for 3D reconstruction
- `thickness_values_<idx>.txt`: Thickness measurements across all slices
- `thickness_measurement_points_<idx>.txt`: 3D vertex indices for thickness measurements

**Benefits:**
- Enables volumetric thickness analysis
- Supports advanced 3D visualizations with proper surface topology
- Creates FreeSurfer-compatible overlay files for integration with other tools

For visualization instructions and outputs, see the [cc_visualization.py documentation](../../scripts/cc_visualization.rst).

### 2D Analysis

When using `--slice_selection middle` or a specific slice number with `--save_template_dir`:

```bash
# Generate 2D template data (middle slice)
python3 fastsurfer_cc.py --sd /data/subjects --sid sub001 \
    --slice_selection middle \
    --save_template_dir /data/templates/sub001
```

**Benefits:**
- Faster processing for single-slice analysis
- 2D visualization is most suitable for displaying downstream statistics
- Compatibility with classical corpus callosum studies

For 2D visualization instructions and outputs, see the [cc_visualization.py documentation](../../scripts/cc_visualization.rst).

### Choosing Analysis Mode

**Use 3D Analysis (`--slice_selection all`) when:**
- You need complete volumetric analysis
- Surface-based visualization is required
- Integration with FreeSurfer workflows is needed
- Comprehensive thickness mapping across the entire corpus callosum is desired

**Use 2D Analysis (`--slice_selection middle` or specific slice) when:**
- Traditional single-slice morphometry is sufficient
- Faster processing is preferred
- Focus is on mid-sagittal cross-sectional measurements
- Compatibility with classical corpus callosum studies is needed

**Note:** The default behavior is `--slice_selection all` for comprehensive 3D analysis. Use `--slice_selection middle` to process only the middle slice for faster, traditional 2D analysis.

## JSON Output Structure

The pipeline generates two main JSON files with detailed measurements and analysis results:

### `stats/callosum.CC.midslice.json` (Middle Slice Analysis)

This file contains measurements from the middle sagittal slice and includes:

**Shape Measurements (single values):**
- `total_area`: Total corpus callosum area (mm²)
- `total_perimeter`: Total perimeter length (mm)
- `circularity`: Shape circularity measure (4π × area / perimeter²)
- `cc_index`: Corpus callosum shape index (length/width ratio)
- `midline_length`: Length along the corpus callosum midline (mm)
- `curvature`: Average curve of the midline (degrees), measured by angle between it's sub-segements

**Subdivisions**
- `areas`: Areas of CC using an improved Hofer-Frahm sub-division method (mm²). This gives more consistent sub-segemnts while preserving the original ratios.

**Thickness Analysis:**
- `thickness`: Average corpus callosum thickness (mm)
- `thickness_profile`: Thickness profile (mm) of the corpus callosum slice (100 thickness values by default, listed from anterior to posterior CC ends) 


**Volume Measurements (when multiple slices processed):**
- `cc_5mm_volume`: Total CC volume within 5mm slab using voxel counting (mm³)
- `cc_5mm_volume_pv_corrected`: Volume with partial volume correction using CC contours (mm³)

**Anatomical Landmarks:**
- `ac_center`: Anterior commissure coordinates in original image space
- `pc_center`: Posterior commissure coordinates in original image space
- `ac_center_oriented_volume`: AC coordinates in standardized space (orient_volume.lta) 
- `pc_center_oriented_volume`: PC coordinates in standardized space (orient_volume.lta)
- `ac_center_upright`: AC coordinates in upright space (cc_up.lta)
- `pc_center_upright`: PC coordinates in upright space (cc_up.lta)

### `stats/callosum.CC.all_slices.json` (Multi-Slice Analysis)

This file contains comprehensive per-slice analysis when using `--slice_selection all`:

**Global Parameters:**
- `slices_in_segmentation`: Total number of slices in the segmentation volume
- `voxel_size`: Voxel dimensions [x, y, z] in mm
- `subdivision_method`: Method used for anatomical subdivision
- `num_thickness_points`: Number of points used for thickness estimation
- `subdivision_ratios`: Subdivision fractions used for regional analysis
- `contour_smoothing`: Gaussian sigma used for contour smoothing
- `slice_selection`: Slice selection mode used

**Per-Slice Data (`slices` array):**

Each slice entry contains the shape measurements, thickness analysis and sub-divisions as described above.
