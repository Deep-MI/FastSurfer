git hook setup / CD
===================

The FastSurfer team has developed a pre-commit hook script to help implement
[Continuous Development and Testing](https://en.wikipedia.org/wiki/Continuous_testing).
This CI/CD expands on github workflows executing them locally. They require a local
[uv installation](https://docs.astral.sh/uv/getting-started/installation/) as described for FastSurfer's
[Native installation](../overview/INSTALL.md#native-ubuntu).

Pre-commit Hook
---------------
The pre-commit hook script will (ignoring files that are git-ignored):
1. Check for non-ASCII file names being added (allow them with `git config hooks.allownonascii true`)
2. Check for trailing white spaces in files
3. Run the lint tests in `test/lint`, like the code-style workflow
4. Sync the uv environment with the `style` and `doc` extras (`uv sync --inexact`); the following checks start
   once it is in sync
5. Run ruff to verify python code formatting is valid
6. Run codespell to check the spelling
7. Run the unit tests in `test/image`, `test/shell` and `test/utils`, like the unittest workflow, but in 4
   (optimal choice, 09-2026) parallel pytest-xdist workers (pytest and pytest-xdist are added with
   `uv run --with`)
8. Run sphinx-build to rebuild the documentation into `FastSurfer/doc-build`.

   Here, one important caveat for documentation editors is that sphinx-build may fail if the documentation file
   structure is changed, without first cleaning the autosummary/autodoc-generated files. To do this, delete the following
   directory `FastSurfer/doc/api/generated`.

The checks run in parallel. A passed check prints one line; the output of a failed check is printed as one block
when it finishes. The output of all checks is saved in `$TMPDIR/fastsurfer-pre-commit-<uid>-<checksum>.log`, the
summary names the file, and the next run overwrites it. The summary lists the status and run time of every check.
All checks run to completion, and any failed check blocks the commit; `git commit --no-verify` skips the hook.
Ctrl-C stops all running checks.

### Installation
To install the pre-commit hook, in the FastSurfer directory call
```bash
ln -s ../../tools/git-hooks/pre-commit .git/hooks/pre-commit
```
