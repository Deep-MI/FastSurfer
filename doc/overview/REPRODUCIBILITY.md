Reproducibility
===============

This page describes which FastSurfer results stay identical between two runs, which do not, and
how to check two runs against each other.

On one machine
--------------

Re-running the same input on the same machine, with the same flags, the same thread count and the
same FastSurfer and FreeSurfer versions, is expected to give the same result.

The topology correction repairs defects in an order that depends on its threads, so it always runs
single-threaded, whatever `--threads` asks for.

With a different thread count
-----------------------------

The thread count is part of what has to stay the same. If it changes:

- the segmentations from the networks stay the same,
- the surface pipeline gives the same surfaces from the same input,
- the bias field correction (N4) does not. It adds up partial results per thread, so its output,
  `mri/orig_nu.mgz`, can differ by one intensity level in a few voxels. The statistics measured on
  that image, such as `stats/aseg+DKT.stats`, `stats/aseg.VINN.stats` and
  `stats/cerebellum.CerebNet.stats`, inherit the difference in their last digits. So can the
  surfaces of a full run, which start from that image.

When you compare runs, pass the same `--threads` to all of them rather than relying on the default.
`--threads 1` keeps every step single-threaded, and `--threads 1 --parallel` does the same while
still processing the two hemispheres at the same time.

Across machines
---------------

Numerical libraries choose their kernels from what the processor offers, so the same code can take
a different path on a different machine. Two layers do this independently, and both have to be
addressed:

- **torch**, in the segmentation networks. The vector instruction set decides the convolution
  kernel, and within one instruction set the vendor can still decide the code branch. Capping the
  instruction set alone leaves a difference between vendors; fixing the branch alone leaves a
  difference between instruction sets. Both are needed.
- **LAPACK and BLAS**, through numpy, in the surface pipeline and the registrations. These pick
  kernels the same way, independently of torch. A difference here is small, but the topology
  correction turns a rounding difference into a different retessellation that every later surface
  inherits, which is how a change far below single precision becomes a visible one.

Only the second of these is pinned for you. The spherical projection, the talairach registration
and the corpus callosum step set the numpy and OpenBLAS variables themselves, because that is where
a rounding difference was found to become a structural one. Nothing sets the torch variables: they
would slow every segmentation down for everyone, and most runs do not need to match another
machine.

**So a default run is not reproducible across machines.** If you need that, set all of them
yourself for the whole run, together with a fixed `--threads`:

```bash
export FASTSURFER_HOME=${FASTSURFER_HOME:-/path/to/FastSurfer}
# the numpy and OpenBLAS values depend on the host, so read them from the helper
eval "$(python $FASTSURFER_HOME/recon_surf/pin_cpu_dispatch.py)"
export ATEN_CPU_CAPABILITY=avx2 ONEDNN_MAX_CPU_ISA=AVX2 MKL_CBWR=COMPATIBLE
```

Pass them into the container with `--env` if you run FastSurfer with docker or singularity. This is
the configuration we test, and with it two x86-64 machines of different vendors produce identical
output. The cost is some speed, because the faster kernels are the ones being declined.

**It also covers the places we found, not every place that could exist.** Any numpy or BLAS call in
a step we have not examined is still free to dispatch on the hardware, which is the other reason to
set these globally rather than to rely on the three steps that pin themselves.

**GPU is not covered.** None of this applies to `--device cuda`: the card model and the precision
modes it selects are a separate source, and we have not tested it. A CPU run and a GPU run of the
same input are not expected to agree, so `--device` is part of what you have to hold constant.

Use one container image as well, see [Running FastSurfer in a container](CONTAINERS.md). On macOS the FreeSurfer
binaries are built without OpenMP and run single-threaded regardless, and that combination is
untested for cross-machine agreement.

Checking a run
--------------

To see which kernels and threads a run used, look at the host block near the top of
`scripts/deep-seg.log` and `scripts/recon-surf.log`. It records the CPU, the instruction set torch
selected, the thread limits and dispatch overrides in force, and a numerical fingerprint. The
fingerprint is the reliable key for "was this comparable hardware": the CPU model name is not,
because the same model can expose different features on different hosts. Each pipeline also logs
the thread budget it used and where that came from.

To compare two runs, use `tools/compare_subjects.py`: it compares voxels, vertices, transforms,
statistics and labels rather than raw bytes, and reports how large each difference is. `diff` and
checksums are no use here, because the headers record timestamps and the command line, so identical
runs still differ byte for byte.

```bash
export FASTSURFER_HOME=${FASTSURFER_HOME:-/path/to/FastSurfer}
export SUBJECTS_DIR=$HOME/my_fastsurfer_analysis
python $FASTSURFER_HOME/tools/compare_subjects.py $SUBJECTS_DIR/subject_a \
    $SUBJECTS_DIR/subject_b
```
