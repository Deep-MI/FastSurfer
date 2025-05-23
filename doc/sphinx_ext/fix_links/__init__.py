from pathlib import Path

from sphinx.application import Sphinx
from sphinx.directives.other import Include
from docutils.parsers.rst import directives

from fix_links.resolve import MySTReplaceDomain, resolve_xref
from fix_links.parser import Parser, wrap_include_run


def setup(app: Sphinx):

    app.add_config_value("fix_links_types", ("ref", "myst", "doc"), "env", list)
    app.add_config_value("fix_links_target", {}, "env", dict)
    app.add_config_value("fix_links_alternative_targets", {}, "env", dict)
    app.add_config_value("fix_links_project_root", Path("."), "env", Path)
    app.add_domain(MySTReplaceDomain)
    app.connect("missing-reference", resolve_xref)

    # override the myst parser without loading the myst parser for default parsing
    # [Sphinx](https://github.com/sphinx-doc/sphinx) extension
    app.add_source_parser(Parser, override=True)

    # update the Include directive's run command
    Include.run = wrap_include_run(Include.run)
    Include.option_spec["relative-images"] = directives.flag
    Include.option_spec["relative-docs"] = directives.path
    Include.option_spec["heading-offset"] = directives.nonnegative_int

    return {"parallel_read_safe": True, "parallel_write_safe": True, "version": "0.1"}
