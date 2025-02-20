---
name: Bug Report
about: You observed a behavior of FastSurfer that is clearly not intended, i.e. not working as intended.
title: ''
labels: bug
assignees: ''

---
Bug Report
==========
<!-- 
Please take a look at the contribution guide https://deep-mi.org/FastSurfer/dev/overview/CONTRIBUTING.html

This issue type is for:
You observed a behavior of FastSurfer that is clearly not intended, i.e. not working as intended.

Please provide clear and concise description of:
-->

Description
-----------
Which functionality, script, interface or interaction does this bug concern? 

Expected Behavior
-----------------
Please provide a clear and concise description of what you expected to happen and/or how this is different to what you 
expected. This also important to determine, if your expectation is also FastSurfer's intention.

Steps to Reproduce
------------------
How can others reproduce this behavior? Please provide a step-by-step guide! If applicable provide error messages, stack
traces, and/or code snippets here. Make sure to **include the command causing the observed behavior** here!

1. Go to '...'
2. Checkout version '...'
3. Run '...'
4. The observed behavior/error is '...'

This is very important to understand and repair the behavior, bug, etc. Best, copy-paste your code or command line call,
but it is also completely fine to use screenshots.

Log Files / Screenshots
-----------------------
Please attach your log files here. They help understanding problems, FastSurfer version, your recognized hardware, etc.

If you are interacting with the standard entrypoint scripts, the relevant log files are:
* `run_fastsurfer.sh`: Log files `$SUBJECTS_DIR/$SUBJECT_ID/scripts/deep-seg.log`, 
  `$SUBJECTS_DIR/$SUBJECT_ID/scripts/BUILD.log` and `$SUBJECTS_DIR/$SUBJECT_ID/scripts/recon-surf.log` (unless use are 
  running with `--seg_only`).
* `long_fastsurfer.sh`: Log files `$SUBJECTS_DIR/$TEMPLATE_ID/scripts/long_fastsurfer.log`.
* `brun_fastsurfer.sh`: Same as `run_fastsurfer.sh`.
* `srun_fastsurfer.sh`: All log files in the work log-directory (`<work>/slurm/logs` and `<work>/slum/scripts`), you may
  choose select one example processing log from there, e.g. `seg_XXXX.log` and `surf_XXXX_Y.log`, try to select a case 
  affected by the bug.

If you can/want to share data privately with the FastSurfer team, you may use the following dropbox:
https://nextcloud.dzne.de/index.php/s/Z2qtHW8c7p3NSJ5

Environment
-----------
Please describe how and which version of FastSurfer you are using. For this fill out the list below!

- **Installation type**: Docker / Singularity Image / native install
- FastSurfer Version: `run_fastsurfer.sh --version all` gives full version information, see also `scripts/BUILD.log`.
- FreeSurfer Version (if you are running a native install): FILL IN HERE
- OS: Linux Version / Windows / Mac
- GPU: e.g. RTX 2080 / ...
<!-- Add any other relevant information on the environment here -->

Additional Context
------------------
Add any other context and comments about the problem here.
