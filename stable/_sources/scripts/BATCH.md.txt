BATCH: brun_fastsurfer.sh
=========================

Usage
-----

```{command-output} ./brun_fastsurfer.sh --help
:cwd: /../
```

Questions
---------
Can I disable the progress bars in the output?

> You can disable the progress bars by setting the TQDM_DISABLE environment variable to 1, if you have tqdm>=4.66.
> 
> For docker, this can be done with the flag `-e`, e.g. `docker run -e TQDM_DISABLE=1 ...`, for singularity with the flag `--env`, e.g. `singularity exec --env TQDM_DISABLE=1 ...` and for native installations by prepending, e.g. `TQDM_DISABLE=1 ./run_fastsurfer.sh ...`.

