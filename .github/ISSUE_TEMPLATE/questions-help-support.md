---
name: Questions/Help/Support
about: You need help in using FastSurfer, for example in interpreting an error message.
title: ''
labels: question
assignees: ''
---
General Support Request
=======================
<!-- 
Please take a look at the contribution guide https://deep-mi.org/FastSurfer/dev/overview/CONTRIBUTING.html

This issue type is for:
You need help in using FastSurfer, for example in interpreting an error message.

Please provide clear and concise description of:
-->

Description
-----------
What are you trying to achieve? 

Steps that lead to your Issue
-----------------------------
Did you find an example/solution in the documentation/external guides that you are following? Please copy-paste a link
or the text of these instructions.

If your results did not match your expectations, please provide a step-by-step guide to reproduce these results and 
state your expectations! 
If applicable provide error messages, stack traces, and/or code snippets here. Make sure to **include the 
commands causing the observed behavior** here, i.e. the commands that brought you to your question!

1. Go to '...'
2. Checkout version '...'
3. Run '....'

This is very important to understand where you are and how to help. Best, copy-paste your code or command line call,
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

- **Installation type**: official docker image / custom docker image / singularity / native
- **FastSurfer version**: `run_fastsurfer.sh --version all` gives full version information, see also `scripts/BUILD.log`.
- **FreeSurfer version** (if you are running a native install): 7.4.1 / 7.3.2 / ...
- **OS**: Linux (Ubuntu) / Windows / Mac / ...
- **GPU**: none / RTX 2080 / ...
<!-- Add any other relevant information on the environment here -->

Additional Context
------------------
Add any other context and comments about the problem here.
