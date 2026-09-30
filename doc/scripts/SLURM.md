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
      ```text
      $ squeue -u $USER --Format JobArrayID,Name,State,Dependency
      1750814_3           FastSurfer-Seg-kueglRUNNING             (null)
      1750815_3           FastSurfer-Surf-kuegPENDING             aftercorr:1750814_*(
      1750816             FastSurfer-Cleanup-kPENDING             afterany:1750815_*(u
      1750815_1           FastSurfer-Surf-kuegRUNNING             (null)
      1750815_2           FastSurfer-Surf-kuegRUNNING             (null)
      ```
      Here, jobs are not finished yet. The FastSurfer-Cleanup-$USER Job moves data to the subject directory (`--sd`).
   2. Check whether there are subject folders and log files in the subject directory, `<subjects_dir>/slurm/logs` for
      the latter.
   3. Check the subject_success file in `<subjects_dir>/slurm/scripts`. It should have a line for each subject for
      both parts of the FastSurfer pipeline, e.g. `<subject_id>: Finished --seg_only successfully` or
      `<subject_id>: Finished --surf_only successfully`! If one of these is missing, the job was likely killed by slurm
      (e.g. because of the time or the memory limit).
   4. For subjects that were unsuccessful (The subject_success will say so), check
      `<subject_dir>/scripts/deep-seg.log` and
      `<subject_dir>/scripts/recon-surf.log` to see what failed.
      Can be found by looking for `"<subject_id>: Failed --seg_only with exit code <return_code>"` (or `--surf_only`) in
      `<subjects_dir>/slurm/scripts/subject_success`.
   5. For subjects that were terminated (missing in subject_success), find which job is associated with subject id
      `grep "<subject_id>" slurm/logs/surf_*.log`, then look at the end of the job and the job step logs
      (`surf_<job_id>_<array_index>.log` and `surf_<job_id>_<array_index>_<step_id>.log`). If slurm terminated the job, it
      will say so there. You can increase the time and memory budget in `srun_fastsurfer.sh` with `--time` and `--mem`
      flags.

      The following bash code snippet can help identify failed runs.
      ```bash
      cd $HOME/my_fastsurfer_analysis
      for sub in *
      do
        if [[ -z "$(grep "$sub: Finished --surf" slurm/scripts/subject_success)" ]]
        then
          echo "$sub was terminated externally"
        fi
      done
      ```
