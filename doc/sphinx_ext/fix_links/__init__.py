

from sphinx.application import Sphinx

from fix_links.resolve import MySTReplaceDomain, resolve_xref


def setup(app: Sphinx):
    app.add_config_value("fix_links_types", ("ref", "myst"), "env", list)
    app.add_config_value("fix_links_target", {}, "env", dict)
    app.add_config_value("fix_links_alternative_targets", {}, "env", dict)
    app.add_domain(MySTReplaceDomain)
    app.connect("missing-reference", resolve_xref)
