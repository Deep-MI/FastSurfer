# Corpus Callosum Pipeline

A deep learning-based pipeline for automated segmentation, analysis, and shape analysis of the corpus callosum in brain MRI scans.
Also segments the fornix, localizes the AC and PC and standardizes the orientation of the brain.

## Overview

This pipeline combines localization and segmentation deep learning models to:
1. Detect AC (Anterior Commissure) and PC (Posterior Commissure) points
2. Extract and align midplane slices
3. Segment the corpus callosum
4. Perform advanced morphometry for corpus callosum, including subdivision, thickness analysis, and various shape metrics
5. Generate visualizations and measurements


## Quickstart

``` python3 fastsurfer_cc.py --subject_dir /path/to/fastsurfer/output --verbose ```

Gives all standard outputs. Then corpus callosum morphometry can be found at `stats/callosum.CC.midslice.json`, including 100 thickness measurements and areas of sub-segments.
Visualization will be placed in `/path/to/fastsurfer/output/qc_snapshots`. For more detailed info see the following sections.

## Command Line Interfaces

### Main Pipeline: `fastsurfer_cc.py`

The main pipeline script performs the complete corpus callosum analysis workflow.

#### Basic Usage

```bash
# Using individual file paths
python3 fastsurfer_cc.py --in_mri /path/to/input/mri.mgz --aseg /path/to/input/aseg.mgz --output_dir /path/to/output --verbose

# Using FastSurfer/FreeSurfer subject directory structure
python3 fastsurfer_cc.py --subject_dir /path/to/fastsurfer/output --verbose
```

#### Required Arguments

Choose one of these input methods:

**Option 1: Individual files**
- `--in_mri PATH`: Input MRI file path (FreeSurfer-conformed)
- `--aseg PATH`: Input segmentation file path
- `--output_dir PATH`: Directory for output files

**Option 2: FastSurfer/FreeSurfer subject directory**
- `--subject_dir PATH`: Subject directory containing standard FastSurfer structure
  - Automatically uses `mri/orig.mgz` and `mri/aparc.DKTatlas+aseg.deep.mgz`
  - Creates standard output paths in FastSurfer structure

#### Optional Arguments

**General Options:**
- `--verbose`: Enable verbose output and debug plots
- `--debug_output_dir PATH`: Directory for debug outputs
- `--cpu`: Force CPU usage even when CUDA is available

**Shape Analysis Parameters:**
- `--num_thickness_points INT`: Number of points for thickness estimation (default: 100)
- `--subdivisions FLOAT [FLOAT ...]`: List of subdivision fractions for CC subsegmentation (default: following Hofer-Frahm definition)
- `--subdivision_method {shape,vertical,angular,eigenvector}`: Method for contour subdivision (default: "shape")
  - `shape`: Intercallosal subdivision perpendicular to intercallosal line
  - `vertical`: Orthogonal to the most anterior and posterior points in AC/PC standardized CC contour
  - `angular`: Subdivision based on equally spaced angles (Hampel et al.)
  - `eigenvector`: Primary direction (same as FreeSurfer's mri_cc)
- `--contour_smoothing FLOAT`: Gaussian sigma for smoothing during contour detection (default: 1.0)
- `--slice_selection {middle,all,INT}`: Which slices to process (default: "all")

**Custom Output Paths:**
- `--upright_volume_path PATH`: Path for upright volume output
- `--segmentation_path PATH`: Path for segmentation output
- `--postproc_results_path PATH`: Path for postprocessing results
- `--cc_markers_path PATH`: Path for CC markers output
- `--upright_lta_path PATH`: Path for upright LTA transform
- `--orient_volume_lta_path PATH`: Path for orientation volume LTA transform
- `--orig_space_segmentation_path PATH`: Path for segmentation in original space
- `--qc_image_path PATH`: Path for QC visualization image

**Template Saving:**
- `--save_template PATH`: Directory path to save contours.txt and thickness_values.txt files

#### Examples

```bash
# Basic analysis with FreeSurfer subject directory
python3 fastsurfer_cc.py --subject_dir /data/subjects/sub001 --verbose

# Custom shape analysis parameters
python3 fastsurfer_cc.py --subject_dir /data/subjects/sub001 \
    --num_thickness_points 150 \
    --subdivisions 0.2 0.4 0.6 0.8 \
    --subdivision_method angular \
    --contour_smoothing 1.5

# Process all slices instead of just middle slice
python3 fastsurfer_cc.py --subject_dir /data/subjects/sub001 \
    --slice_selection all

# Save template files for visualization
python3 fastsurfer_cc.py --subject_dir /data/subjects/sub001 \
    --save_template /data/templates/sub001
```

## Outputs

The pipeline produces the following outputs in the specified output directory:

### Main Pipeline Outputs

**Analysis Results:**
- `stats/callosum.CC.midslice.json`: Contains detected landmarks and measurements for the middle slice
- `stats/callosum.CC.all_slices.json`: Enhanced postprocessing results with per-slice analysis

**Transformation Matrices:**
- `mri/transforms/cc_up.lta`: Transformation from original to upright space (aligned to fsaverage, CC midslice at the center)
- `mri/transforms/orient_volume.lta`: Transformation a CC, AC & PC standardized space. The CC is at the center and AC & PC on the coordinate line, standardizing the head orientation.

**Image Volumes:**
- `mri/callosum_seg_upright.mgz`: Corpus callosum segmentation in upright space (aligned to fsaverage, matching cc_up.lta)
- `mri/callosum_seg_aseg_space.mgz`: Corpus callosum segmentation in conformed image orientation (aligned to orig.mgz and other segmentations)
- `mri/callosum_seg_soft.mgz`: Corpus callosum soft labels (segmentation probabilities, upright space)
- `mri/fornix_seg_soft.mgz`: Fornix soft labels (segmentation probabilities, upright space)
- `mri/background_seg_soft.mgz`: Background soft labels (segmentation probabilities, upright space)


**Quality Control and Visualizations:**
- `qc_snapshots/callosum.png`: Debug visualization of corpus callosum contours and thickness measurements
- `qc_snapshots/callosum_thickness.png`: 3D thickness visualization (when using `--slice_selection all`)
- `qc_snapshots/corpus_callosum.html`: Interactive 3D mesh visualization (when using `--slice_selection all`)


**Surface Files (only provided when using `--slice_selection all`):**
- `surf/callosum.surf`: FreeSurfer surface format for integration with FreeSurfer tools (e.g. freeview)
- `surf/callosum.thickness.w`: FreeSurfer overlay file containing thickness values
- `surf/callosum_mesh.vtk`: VTK format mesh file for 3D visualization 

**Template Files (when --save_template is used):**

- `contours.txt`: Corpus callosum contour coordinates for visualization
- `thickness_values.txt`: Thickness measurements at each contour point
- `measurement_points.txt`: Original vertex indices where thickness was measured

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




## Visualization: `cc_visualization.py`

Creates advanced visualizations of corpus callosum from template files generated by the main pipeline.
Useful for visualization of analysis results.

#### Basic Usage

```bash
# Using contours file
python3 cc_visualization.py --contours /path/to/contours.txt \
    --thickness /path/to/thickness_values.txt \
    --measurement_points /path/to/measurement_points.txt \
    --output_dir /path/to/output

# Using fsaverage template (no contours file)
python3 cc_visualization.py \
    --thickness /path/to/thickness_values.txt \
    --measurement_points /path/to/measurement_points.txt \
    --output_dir /path/to/output
```

#### Required Arguments

- `--thickness PATH`: Path to thickness_values.txt file
- `--measurement_points PATH`: Path to measurement points file containing original vertex indices
- `--output_dir PATH`: Directory for output files

#### Optional Arguments

**Input:**
- `--contours PATH`: Path to contours.txt file (if not provided, uses fsaverage template)

**Mesh Parameters:**
- `--resolution FLOAT`: Resolution in mm for the mesh (default: 1.0)
- `--smooth_iterations INT`: Number of smoothing iterations to apply to the mesh (default: 1)

**Visualization Options:**
- `--colormap {red_to_blue,blue_to_red,red_to_yellow,yellow_to_red}`: Colormap for thickness visualization (default: "red_to_yellow")
- `--color_range MIN MAX`: Optional fixed range for the colorbar
- `--legend STRING`: Legend for the colorbar (default: "Thickness (mm)")
- `--twoD`: Generate 2D visualization instead of 3D mesh

#### Colormap Options

- `red_to_blue`: Red → Orange → Grey → Light Blue → Blue
- `blue_to_red`: Blue → Light Blue → Grey → Orange → Red  
- `red_to_yellow`: Red → Yellow → Light Blue → Blue
- `yellow_to_red`: Yellow → Light Blue → Blue → Red

#### Examples

```bash
# Basic 3D mesh visualization
python3 cc_visualization.py \
    --thickness /data/templates/sub001/thickness_values.txt \
    --measurement_points /data/templates/sub001/measurement_points.txt \
    --output_dir /data/visualizations/sub001

# 2D visualization with custom colormap
python3 cc_visualization.py \
    --thickness /data/templates/sub001/thickness_values.txt \
    --measurement_points /data/templates/sub001/measurement_points.txt \
    --output_dir /data/visualizations/sub001 \
    --twoD \
    --colormap blue_to_red
```

## Analysis and Visualization Workflow

The pipeline supports different analysis modes that determine the type of template data generated and corresponding visualization options:

### 3D Analysis and Visualization

When running the main pipeline with `--slice_selection all` and `--save_template`, a complete 3D template is generated:

```bash
# Generate 3D template data
python3 fastsurfer_cc.py --subject_dir /data/subjects/sub001 \
    --slice_selection all \
    --save_template /data/templates/sub001
```

This creates:
- `contours.txt`: Multi-slice contour data for 3D reconstruction
- `thickness_values.txt`: Thickness measurements across all slices
- `measurement_points.txt`: 3D vertex indices for thickness measurements

The 3D template can then be visualized using the standard 3D mesh options:

```bash
# Create 3D mesh visualization
python3 cc_visualization.py \
    --contours /data/templates/sub001/contours.txt \
    --thickness /data/templates/sub001/thickness_values.txt \
    --measurement_points /data/templates/sub001/measurement_points.txt \
    --output_dir /data/visualizations/sub001
```

**3D Analysis Benefits:**
- Generates complete surface meshes (VTK, FreeSurfer formats)
- Enables volumetric thickness analysis
- Supports advanced 3D visualizations with proper surface topology
- Creates FreeSurfer-compatible overlay files for integration with other tools

### 2D Analysis and Visualization

When using `--slice_selection middle` or a specific slice number with `--save_template`:

```bash
# Generate 2D template data (middle slice)
python3 fastsurfer_cc.py --subject_dir /data/subjects/sub001 \
    --slice_selection middle \
    --save_template /data/templates/sub001

# Or specific slice
python3 fastsurfer_cc.py --subject_dir /data/subjects/sub001 \
    --slice_selection 5 \
    --save_template /data/templates/sub001
```

This creates template data for a single slice, which should be visualized in 2D mode:

```bash
# Create 2D visualization
python3 cc_visualization.py \
    --thickness /data/templates/sub001/thickness_values.txt \
    --measurement_points /data/templates/sub001/measurement_points.txt \
    --output_dir /data/visualizations/sub001 \
    --twoD
```

**2D Analysis Benefits:**
- Faster processing for single-slice analysis
- 2D visualization is most suitable for displaying downstream statistics

### Surface Generation Requirements

**Important:** Complete surface files (VTK, FreeSurfer surface formats, overlay files) are only generated when using `--slice_selection all`. Single-slice analysis cannot produce proper 3D surface topology and will not generate these files.

**3D Surface Outputs (only with `--slice_selection all`):**
- `cc_mesh.vtk`: Complete 3D surface mesh
- `cc_mesh.fssurf`: FreeSurfer surface format
- `cc_mesh_overlay.curv`: Thickness overlay for FreeSurfer visualization

**2D Outputs (any slice selection):**
- `cc_mesh_snap.png`: 2D visualization or 3D mesh snapshot
- Standard analysis JSON files with measurements

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



## Visualization Tool Outputs

When using `cc_visualization.py`, additional outputs are generated (for advanced users).

**3D Mode Outputs (default):**
- `cc_mesh.vtk`: VTK format mesh file for 3D visualization
- `cc_mesh.fssurf`: FreeSurfer surface format
- `cc_mesh_overlay.curv`: FreeSurfer overlay file with thickness values
- `cc_mesh.html`: Interactive 3D mesh visualization
- `cc_mesh_snap.png`: Snapshot image of the 3D mesh
- `midslice_2d.png`: 2D visualization of the middle slice

**2D Mode Outputs (when `--twoD` is specified):**
- `cc_thickness_2d.png`: 2D contour visualization with thickness colormap