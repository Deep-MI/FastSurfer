# Instructions for agents working on the documentation

- Follow [CONVENTIONS.md](CONVENTIONS.md) when writing or changing documentation: the Markdown and reST files in
  `doc/` and the files they include with `.. include::` (for example `README.md` and `tools/Docker/README.md`).
- `AGENTS.md` and `CONVENTIONS.md` are not part of the built documentation, `exclude_patterns` in `doc/conf.py` lists
  them.
- Markdown is parsed by the `fix_links` extension (`doc/sphinx_ext/fix_links`), which extends MyST: among others, it
  substitutes `{{ NAME }}` in code and renders banners on top of code blocks, see CONVENTIONS.md.
- Check changes by building the documentation from the repository root, warnings are errors:

  ```bash
  uv run --extra doc sphinx-build -WT --keep-going -j auto doc doc-build
  ```
