SLURM: srun_fastsurfer.sh
=========================

Usage
-----

```{command-output} ./srun_fastsurfer.sh --help
:cwd: /../
```

Debugging SLURM runs
--------------------

1. Did the run succeed?

   1. Check whether all jobs are done (specifically the copy job).
      ```bash
      $ squeue -u $USER --Format JobArrayID,Name,State,Dependency
      1750814_3           FastSurfer-Seg-kueglRUNNING             (null)
      1750815_3           FastSurfer-Surf-kuegPENDING             aftercorr:1750814_*(
      1750816             FastSurfer-Cleanup-kPENDING             afterany:1750815_*(u
      1750815_1           FastSurfer-Surf-kuegRUNNING             (null)
      1750815_2           FastSurfer-Surf-kuegRUNNING             (null)
      ```
      Here, jobs are not finished yet. The FastSurfer-Cleanup-$USER Job moves data to the subject directory (--sd).

   2. Check whether there are subject folders and log files in the subject directory, <subject directory>/slurm/logs for the latter.

   3. Check the subject_success file in <subject directory>/slurm/scripts. It should have a line for each subject for both parts of the FastSurfer pipeline, e.g. `<subject id>: Finished --seg_only successfully` or `<subject id>: Finished --surf_only successfully`! If one of these is missing, the job was likely killed by slurm (e.g. because of the time or the memory limit).

   4. For subjects that were unsuccessful (The subject_success will say so), check `<subject directory>/<subject id>/scripts/deep-seg.log` and `<subject directory>/<subject id>/scripts/recon-surf.log` to see what failed.
      Can be found by looking for "<subject id>: Failed <--seg_only/--surf_only> with exit code <return code>" in `<subject directory>/slurm/scripts/subject_success`.

   5. For subjects that were terminated (missing in subject_success), find which job is associated with subject id `grep "<subject id>" slurm/logs/surf_*.log`, then look at the end of the job and the job step logs (surf_XXX_YY.log and surf_XXX_YY_ZZ.log). If slurm terminated the job, it will say so there. You can increase the time and memory budget in `srun_fastsurfer.sh` with `--time` and `--mem` flags.
      The following bash code snippet can help identify failed runs.
      ```
      cd <subject directory>
      for sub in *
      do
      if [[ -z "$(grep "$sub: Finished --surf" slurm/scripts/subject_success)" ]]
      then
          echo "$sub was terminated externally"
      fi
      done
      ```
