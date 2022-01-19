# Utilities

This directory contains some useful utility scripts.

## Command Time Extraction

The `extract_recon_surf_time_info.py` script can be used to generate a yaml file containing information on the commands executed in recon_surf from a `recon-surf.log` file.
Every command has a corresponding entry, which includes the information:
* cmd_name: the full command
* start: the time at which the command was executed
* stop: the time at which command execution terminated
* duration: the time elapsed between *start* and *stop* (in minutes, by default)

Entries are grouped according to the section in `recon_surf.sh` in which the commands are executed.
`recon-all` commands are further divided into individual stages.

### Arguments
* `--input_file_path`: Absolute path to the input `recon-surf.log` file. By default, this is `scripts/recon-surf.log` in the current directory.
* `--output_file_path`: Absolute path to the output `recon-surf_times.yaml` file. By default, the file will be saved in the same directory as the input file.
* `--time_units`: Units for duration: s (seconds) or m (minutes; default)

### Example

The following will extract recon_surf command time information from `123456/scripts/recon-surf.log` and save it in `123456/scripts/recon-surf_times.yaml` (with durations in minutes).

```
python3 extract_recon_surf_time_info.py --input_file_path 123456/scripts/recon-surf.log --time_units m
```

