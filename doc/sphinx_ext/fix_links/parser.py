from functools import wraps
from os.path import relpath
from pathlib import Path
from typing import Optional, cast
from itertools import chain

from docutils import nodes
from markdown_it import MarkdownIt
from markdown_it.tree import SyntaxTreeNode
from myst_parser.mdit_to_docutils.sphinx_ import SphinxRenderer
from myst_parser.sphinx_ import Parser as MySTParser
from sphinx import addnodes
from sphinx.directives.other import Include


def wrap_include_run(method):
    @wraps(method)
    def _wrapper(include_instance: Include):
        doc_settings = include_instance.state.document.settings
        key = "fix_links_parser_options"
        if hasattr(doc_settings, key):
            options = getattr(doc_settings, key, {})
        else:
            options = {}
            setattr(doc_settings, key, options)

        source_dir = Path(include_instance.state.document["source"]).parent
        include_path = (source_dir / include_instance.arguments[0]).resolve()
        if "relative-images" in include_instance.options:
            from os.path import relpath
            options["relative-images"] = relpath(include_path.parent, source_dir)
        relative_docs = include_instance.options.get("relative-docs", ".")
        if relative_docs != "/":
            options["relative-docs"] = (relative_docs, source_dir, include_path.parent)
        return method(include_instance)

    return _wrapper


class Renderer(SphinxRenderer):
    """
    Renderer object to automatically fix headings that are not consecutive levels in
    (included) Markdown files. Also includes alternative targets into anchors that
    are rendered, but do not match a target.
    """

    def __init__(self, parser: MarkdownIt):
        self._heading_base: Optional[int] = None
        super().__init__(parser)

    def update_section_level_state(self, section: nodes.section, level: int) -> None:
        """This method is fixed such that """
        parent_level = max(
            section_level
            for section_level in self._level_to_section
            if level > section_level
        )
        if self._heading_base is None:
            if (level > parent_level) and (parent_level + 1 != level):
                self._heading_base = level - parent_level - 1
            else:
                self._heading_base = 0

        new_level = level - self._heading_base
        if new_level < 0:
            msg = (f"We fixed the offset to {new_level} based on the first heading, "
                   f"but following headings have lower numbers")
            from myst_parser.warnings_ import MystWarnings
            self.create_warning(
                msg,
                MystWarnings.MD_HEADING_NON_CONSECUTIVE,
                line=section.line,
                append_to=self.current_node,
            )
            self._heading_base = level
            new_level = 0

        super().update_section_level_state(section, new_level)

    def _handle_relative_docs(self, destination: str) -> str:
        from os.path import relpath, normpath
        if destination.startswith("/"):
            return relpath(destination[1:], self.sphinx_env.srcdir)
        relative_include = self.md_env.get("relative-docs", None)
        if relative_include is not None:
            source_dir: Path
            source_dir, include_dir = relative_include[1:]
            return relpath(
                include_dir / relative_include[0] / normpath(destination),
                source_dir,
            )
        return destination

    def render_link_anchor(self, token: SyntaxTreeNode, target: str) -> None:

        if not target.startswith("#"):
            return self.render_link_unknown(token)

        if target[1:] in self.document.nameids:
            return super().render_link_anchor(token, target)

        cfg_alt_tgts = self.sphinx_env.config.fix_links_alternative_targets

        include_abspaths = (Path(inc[0]).resolve() for inc in self.document.include_log)
        doc_root = self.sphinx_env.srcdir
        include_relpaths = (f"/{relpath(path, doc_root)}" for path in include_abspaths)
        includes = (".",) + tuple(include_relpaths)
        alt_targets = dict.fromkeys(chain(*(cfg_alt_tgts.get(f, ()) for f in includes)))

        # href_before = token.attrGet("href")
        token.attrs["href"] = Path(self.current_node.source).name + target
        self.render_link_unknown(token)

        ref_node = self.current_node.children[-1]
        if isinstance(ref_node, addnodes.pending_xref):
            ref_node["alternative_targets"] = tuple(alt_targets)

    def render_link_unknown(self, token: SyntaxTreeNode) -> None:
        super().render_link_unknown(token)
        ref_node: nodes.Element = cast(nodes.Element, self.current_node.children[-1])
        attr = ref_node.attributes
        if (attr.get("refdomain", "") == "doc" and
                (target := attr.get("reftarget", "")).startswith("..")):
            attr["refdomain"] = None
            # project_root: how absolute paths are interpreted w.r.t. the doc root
            doc_root = Path(self.sphinx_env.srcdir)
            project_root = self.sphinx_env.config.fix_links_project_root
            target_path = relpath(
                (doc_root / target).resolve(),
                (doc_root / project_root).resolve(),
            )
            attr["reftarget"] = f"/{target_path}"


class Parser(MySTParser):
    """
    Parser to use `Renderer`, which automatically fixes non-consecutive headings and
    manages alternative targets in the topmatter.
    """

    def parse(self, inputstring: str, document: nodes.document) -> None:
        """Parse source text.

        :param inputstring: The source string to parse
        :param document: The root docutils node to add AST elements to

        """
        from myst_parser.warnings_ import create_warning
        from myst_parser.parsers.mdit import create_md_parser
        from myst_parser.config.main import (
            MdParserConfig, TopmatterReadError, merge_file_level, read_topmatter,
        )

        # get the global config
        config: MdParserConfig = document.settings.env.myst_config
        alt_targets = ()

        # update the global config with the file-level config
        try:
            topmatter = read_topmatter(inputstring)
        except TopmatterReadError:
            pass  # this will be reported during the render
        else:
            if topmatter:
                if "alternative-targets" in topmatter:
                    alt_targets = tuple(topmatter.pop("alternative-targets").split())
                warning = lambda wtype, msg: create_warning(  # noqa: E731
                    document, msg, wtype, line=1, append_to=document,
                )
                config = merge_file_level(config, topmatter, warning)

        from contextlib import contextmanager

        @contextmanager
        def _restore(node, cfg_name: str, values: tuple[str]):
            cfg = getattr(node, cfg_name)
            before = cfg.get(".", ())
            cfg["."] = before + values
            yield
            cfg["."] = before

        parser = create_md_parser(config, Renderer)
        with _restore(
                document.settings.env.config,
                "fix_links_alternative_targets",
                alt_targets,
        ):
            parser.options["document"] = document
            parser_options = getattr(document.settings, "fix_links_parser_options", {})
            parser.render(inputstring, parser_options)
