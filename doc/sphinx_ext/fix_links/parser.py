

from typing import Optional
from itertools import chain

from docutils import nodes
from markdown_it import MarkdownIt
from markdown_it.tree import SyntaxTreeNode
from myst_parser.mdit_to_docutils.sphinx_ import SphinxRenderer
from myst_parser.sphinx_ import Parser as MySTParser
from sphinx import addnodes


class Renderer(SphinxRenderer):
    """
    Renderer object to automatically fix headings that are not consequetive levels in
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

    def render_link_anchor(self, token: SyntaxTreeNode, target: str) -> None:

        if not target.startswith("#"):
            return self.render_link_unknown(token)

        if target[1:] in self.document.nameids:
            return super().render_link_anchor(token, target)

        cfg_alt_tgts = self.sphinx_env.config.fix_links_alternative_targets
        includes = (".", *(file for file, _ in self.document.include_log))
        alt_targets = dict.fromkeys(chain(*(cfg_alt_tgts.get(f, ()) for f in includes)))

        # href_before = token.attrGet("href")
        token.attrs["href"] = self.current_node.source + target
        self.render_link_unknown(token)

        ref_node = self.current_node.children[0]
        if isinstance(ref_node, addnodes.pending_xref):
            ref_node["alternative_targets"] = tuple(alt_targets)


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
                    document, msg, wtype, line=1, append_to=document
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
            parser.render(inputstring)
