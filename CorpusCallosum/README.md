# Corpus Callosum Pipeline

A deep learning-based pipeline for automated segmentation, analysis, and shape analysis of the corpus callosum in brain MRI scans.
Also segments the fornix, localizes the anterior and posterior commissure (AC and PC) and standardizes the orientation of the brain.

For detailed documentation, please refer to:
- [Module Overview](../doc/overview/modules/CC.md): Detailed description of the pipeline, workflow, and analysis options.
- [Output Files](../doc/overview/OUTPUT_FILES.md#corpus-callosum-module): List of output files and their descriptions.

## Quickstart

```bash
python3 fastsurfer_cc.py --sd /path/to/fastsurfer/output --sid test-case --verbose
```

Gives all standard outputs. The corpus callosum morphometry can be found at `stats/callosum.CC.midslice.json` including 100 thickness measurements and the areas of sub-segments.
