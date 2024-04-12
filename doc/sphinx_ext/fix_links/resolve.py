
import re
from functools import lru_cache, partial
from pathlib import Path

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
            logger.debug(f"Skipping replacement of {node.attibutes} (no {attr})")
            return
        logger.debug(f"Searching for replacement of {node[attr]}:")
        orig_source = node.attributes[attr]
        source = node.attributes[attr]
        relpath = str(Path(env.srcdir).relative_to(Path.cwd()))
        # TODO fix hardcoded doc here
        for prefix in (str(env.srcdir), relpath, "../", "../doc"):
            if orig_source.startswith(prefix):
                source = orig_source.removeprefix(prefix)
                # maybe this already fixed the path?
                ref = _resolve_xref_with_(source.lower(), orig_source)

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
                            f"Infinite replacement loop with string '{source}', "
                            f"pattern '{key}' and replacement '{repl}'!",
                        )
                        break
                    replaced = _replaced
                # search for a reference associated with the replaced link in std
                ref = _resolve_xref_with_(str(replaced).lower(), orig_source)

                # check and return the reference, if it is valid
                if ref is not None:
                    return ref
            # if the pattern matched, but none of the replacements lead to a valid
            # reference
            logger.warning(
                f"Could not find reference {node['reftarget']} in {node['refdoc']}!"
            )
        # restore the reftarget attribute
        node[attr] = source
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
            f"<{node.source}> replacing {source} with {'='.join(target)}",
            location=node['refdoc'],
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
