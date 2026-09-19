FastSurfer pipeline test
========================

`pipelinetest.yaml` runs the whole FastSurfer pipeline on two sample subjects inside the Docker
image built from the branch under test, compares every output against a stored reference, and grades
the run. It is the end-to-end test: the unit tests in `unittest.yaml` cover the pieces, this covers
what they produce together.

It is expensive, it needs secrets, and it is restricted to `Deep-MI/FastSurfer`.

How it is started
-----------------

**From the Actions tab (`workflow_dispatch`), on any branch.** This tests that branch as it stands
and reports on the run. It is also how reference data is produced: run it on the commit the
reference should come from and keep the resulting subject directories.

Three inputs, all optional:

- `docker-image`: `build-cached` (the default) builds the image from the branch this runs on.
  Anything else is taken as the name of an image to pull instead.
- `freesurfer-build-image`: overrides the FreeSurfer build image, which otherwise comes from
  `tool.freesurfer.version` in `pyproject.toml`.
- `refresh-cache`: resolve the dependencies afresh rather than restoring the cached layers, and
  write the result back for later runs. The weekly scheduled run does this on its own, so reach for
  it when you want a new dependency version tested now rather than on the next weekly run.

**On a nightly schedule, on the default branch.** Most nights stop immediately: the run only
continues if the branch moved since the last one, plus once a week whatever happened. The weekly run
exists because an unchanged branch that produces a changed result means something under us moved,
and a run gated purely on our own commits can never see that. It is also the run that refreshes the
build cache, see below, because otherwise the dependency versions would never move either.

**Not on a pull request.** GitHub passes no secrets to a workflow triggered from a fork, and must
not, because the code under test runs in the same job as the FreeSurfer license. A trigger on a pull
request could therefore never work for a fork, which is where most contributions come from, while
producing a skipped check on every unrelated label on every pull request. To test a branch before it
lands, dispatch on it.

Jobs
----

### docker image build

Decides whether the run applies at all and, when it does, builds the image with
`tools/Docker/build.py`. This is also the job that publishes the outputs the other two read, chiefly
whether to continue and which image to use.

The build is cached into the GitHub Actions cache, so an unchanged source tree restores the layers
rather than rebuilding them. That includes the layer that resolves the Python dependencies, which
sits in an earlier stage than the one copying our source, so committing does not invalidate it and
a new release on PyPI would otherwise stay invisible indefinitely. The weekly run therefore builds
without restoring the cache and writes what it resolved, so the following week starts from that.
By hand, `build.py --refresh_cache` does the same thing.

### run and tests, once per subject

A matrix over the two sample subjects, one at 1.0mm and one at 0.8mm, each with its own sample
image, its own reference, and its own extra arguments. Each entry runs FastSurfer end to end and
then runs the `test/pipelinetest` suite against the reference, which compares image headers, segmentation
overlap, intensity volumes, surface geometry and the stats tables, each to a tolerance recorded in
`test/pipelinetest/data`.

The first step of the job reports the runner CPU. The pipeline is pinned so that results do not
depend on it, and that line is how the pin is checked rather than assumed.

### results annotation

Collects the JUnit files from both subjects into a single `pipelinetest results` check.

Grading
-------

Each subject's run ends with a verdict, in three grades, because drift and breakage call for
different responses:

| grade  | meaning                                                                      | outcome         |
|--------|------------------------------------------------------------------------------|-----------------|
| green  | every compared output is identical to the reference                          | passes quietly  |
| yellow | outputs differ, but all stay within tolerance                                | warning, passes |
| red    | a tolerance was exceeded, a file is missing, or the comparison could not run | fails           |

Yellow is the interesting one. A rerun of unchanged code should be identical, so any difference is
worth a look even when it is inside the limits. Red on a deliberate change, a new module for
instance, means the reference needs rebuilding rather than the tolerance widening.

Secrets
-------

All five are required. They keep an older `QUICKTEST_` prefix because a secret cannot be renamed,
only deleted and recreated.

- `QUICKTEST_IMAGE_HREF_1mm`, `QUICKTEST_IMAGE_HREF_08mm`: the sample images
- `QUICKTEST_TARGET_HREF_1mm`, `QUICKTEST_TARGET_HREF_08mm`: the reference processing of each
- `QUICKTEST_LICENSE`: the contents of a FreeSurfer license file

Running the comparison yourself
-------------------------------

The same suite runs locally against any two subject directories:

```bash
REF_DIR=<reference> SUBJECTS_DIR=<output> FASTSURFER_HOME=$PWD \
  uv run --extra pipelinetest python -m pytest test/pipelinetest
```

`tools/compare_subjects.py` does the coarser file-by-file comparison the grading step uses, and
reports which outputs differ without applying any tolerance.
