git hook setup / CD
===================

The FastSurfer team has developed a pre-commit hook script to help implement
[Continuous Development and Testing](https://en.wikipedia.org/wiki/Continuous_testing).
This CI/CD expands on github workflows executing them locally. They require a local
[uv installation](https://docs.astral.sh/uv/getting-started/installation/) as described for FastSurfer's
[Native installation](../overview/INSTALL.md#native-ubuntu-2004-or-ubuntu-2204).

Pre-commit Hook
---------------
The pre-commit hook script will:
1. Check for trailing white spaces in files
2. Run ruff to verify python code formatting is valid
3. Run codespell to check the spelling
4. Run sphinx-build to rebuild the documentation into `FastSurfer/doc-build`.

   Here, one important caveat for documentation editors is that sphinx-build may fail if the documentation file
   structure is changed, without first cleaning the autosummary/autodoc-generated files. To do this, delete the following
   directory `FastSurfer/doc/api/generated`.

### Installation
To install the pre-commit hook, in the FastSurfer directory call
```bash
ln -s ../../tools/git-hooks/pre-commit .git/hooks/pre-commit
```
