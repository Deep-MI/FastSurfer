
import re
from functools import lru_cache, partial
from pathlib import Path
from typing import Generator, Any

import sphinx.domains
from docutils import nodes
from sphinx.domains import Domain
from sphinx.application import Sphinx
from sphinx.environment import BuildEnvironment
from sphinx.builders import Builder
from sphinx import addnodes
from sphinx.util.logging import getLogger

logger = getLogger(__name__)


@lru_cache
def make_pattern(s: str) -> re.Pattern:
    return re.compile(s, re.IGNORECASE)


def loc(node) -> str:
    return node["refdoc"] if "refdoc" in node.attributes else node.source


def resolve_included(
        included: dict[str, set[str]],
        found_docs: set[str],
        uri_path: str,
) -> str:
    """
    Iterate through including files resolved via inclusion links.

    Parameters
    ----------
    included : dict[str, set[str]]
        The dictionary mapping a file to the files it includes.
    found_docs : set[str]
        A set of doc files that are part of the documentation.
    uri_path : str
        The path to the included file

    Returns
    -------
    str
        The resolved path.
    """
    def __resolve_all(path, include_tree=()):
        for src, inc in included.items():
            if path in inc:
                if src in found_docs:
                    yield src
                elif src in include_tree:
                    logger.warning(f"Recursive inclusion in {path} -> {src}!")
                else:
                    yield from __resolve_all(src, include_tree + (path,))

    yield from __resolve_all(uri_path)


def resolve_xref(
        app: Sphinx,
        env: BuildEnvironment,
        node: addnodes.pending_xref,
        contnode: nodes.Element,
) -> nodes.reference | None:
    """
    Replace unresolved names by replacing the link with configurable alternatives.

    For an unresolved :py:`sphinx.addnodes.pending_xref` `node` of reftype in the
    config variable fix_links_types, this function will look through registered
    replacements and replace with alternative labels. The first successful replacement,
    that also matches a link in the documentation will be returned.

    This function is compatible with the missing-references sphinx-event.

    Parameters
    ----------
    app : sphinx.application.Sphinx
    env : sphinx.environment.BuildEnvironment
    node : sphinx.addnodes.pending_xref
    contnode : docutils.noes.Element

    Returns
    -------
    docutils.nodes.reference, None
        The first node that successfully links to a valid target.
    """
    config = env.config
    attr = "reftarget"
    if node.attributes.get("reftype", "") in config.fix_links_types:
        subs = {k: (make_pattern(k), v) for k, v in config.fix_links_target.items()}
        _resolve_xref_with_ = partial(_resolve_xref_with, app, env, node, contnode)

        if attr not in node.attributes:
            logger.debug(
                f"[fix_links] Skipping replacement of {node.attibutes} (no {attr})",
                location=loc(node),
            )
            return
        logger.debug(
            f"[fix_links] Searching for replacement of {node[attr]}:",
            location=loc(node),
        )

        from os.path import relpath
        # project_root: how absolute paths are interpreted w.r.t. the doc root
        doc_root = Path(env.srcdir)
        project_root = env.config.fix_links_project_root
        uri = node[attr]
        if node["refdomain"] == "doc":
            _uri_path, _uri_id = f"/{uri}", node.attributes.get("refid", None) or ""
            _uri_sep = "#" if _uri_id else ""
            project_root = "."
        else:
            _uri_path, _uri_sep, _uri_id = uri.partition("#")
            if not _uri_id and getattr(node, "reftargetid", None) is not None:
                _uri_sep, _uri_id = "#", node["reftargetid"]
        # resolve the target Path in the link w.r.t. the source it came from
        if _uri_path.startswith("/"):
            # absolute with respect to documentation root
            target_path = (doc_root / project_root / _uri_path[1:]).resolve()
        else:
            sourcefile_path = Path(env.srcdir) / node["refdoc"]
            target_path = (sourcefile_path.parent / _uri_path).resolve()
        _uri_path = relpath(target_path, env.srcdir)
        _uri_hash = _uri_sep + _uri_id

        if not _uri_path.startswith("../"):
            # maybe this already fixed the path?
            ref = _resolve_xref_with_(f"/{_uri_path}{_uri_hash}".lower(), uri)
            if ref is not None:
                return ref

        # trace back the include path and check if this resolves the ref
        if env.included:
            potential_targets = resolve_included(
                env.included,
                env.found_docs,
                _uri_path,
            )
            for potential_doc in potential_targets:
                potential_path = env.doc2path(potential_doc, False)
                ref = _resolve_xref_with_(f"/{potential_path}{_uri_hash}".lower(), uri)
                if ref is not None:
                    return ref

        source = f"/{relpath(target_path, env.srcdir)}{_uri_sep}{_uri_id}"
        for key, (pat, repls) in subs.items():
            # if this search string does not match, try next
            if not pat.match(source):
                continue

            # iterate over different replacement options
            for repl in repls:
                # repeatedly replace until no more changes are occur
                replaced = pat.sub(repl, source)
                while pat.match(replaced):
                    _replaced = pat.sub(repl, replaced)
                    if replaced == _replaced:
                        logger.warning(
                            f"[fix_links] Infinite replacement loop with string "
                            f"'{source}', pattern '{key}' and replacement '{repl}'!",
                            location=loc(node),
                        )
                        break
                    replaced = _replaced
                # search for a reference associated with the replaced link in std
                ref = _resolve_xref_with_(str(replaced).lower(), uri)

                # check and return the reference, if it is valid
                if ref is not None:
                    return ref
            # if the pattern matched, but none of the replacements lead to a valid
            # reference
            logger.warning(
                f"[fix_links] Could not find reference {node['reftarget']}!",
                location=node.source,
            )
        # restore the reftarget attribute
        node[attr] = uri
    # node["reftype"] = prev_type


def _resolve_xref_with(
        app: Sphinx,
        env: BuildEnvironment,
        node: addnodes.pending_xref,
        contnode: nodes.Element,
        target: str,
        source: str,
) -> nodes.reference | None:
    std_domain = env.domains["std"]
    ref: nodes.reference | None = std_domain.resolve_xref(
        env,
        node["refdoc"],  # fromdocname
        app.builder,
        "ref",
        target,
        node,
        contnode,
    )

    # check and return the reference, if it is valid
    if ref is not None:
        attrs = ("reftarget", "refuri", "refid")
        target = next((a, ref[a]) for a in attrs if a in ref.attributes)
        logger.debug(
            f"[fix_links] <{node.source}> replacing {source} with {'='.join(target)}",
            location=loc(node),
        )
    return ref


class MySTReplaceDomain(Domain):
    """"""

    name: str = "myst_repl"

    def resolve_any_xref(
            self,
            env: BuildEnvironment,
            fromdocname: str,
            builder: Builder,
            target: str,
            node: addnodes.pending_xref,
            contnode: nodes.Element,
    ) -> list[tuple[str, nodes.Element]]:
        try:
            ref: nodes.Element | None = resolve_xref(env.app, env, node, contnode)
            if ref is not None:
                return [("std:ref", ref)]
        except StopIteration:
            pass
        return []
