from pprint import pprint
import re
from functools import lru_cache

from docutils import nodes
from sphinx import addnodes
from sphinx.util.logging import getLogger

logger = getLogger(__name__)


@lru_cache
def make_pattern(s: str) -> tuple[str, re.Pattern]:
    if s.startswith("re-") and "=" in s:
        prefix, regex = s.split("=", 2)
        return prefix[3:], re.compile(regex)
    raise ValueError(f"{s} is not valid pattern!")


def is_ref_inline(node):
    return (isinstance(node, nodes.reference) and
            len(node.children) and
            node.children[0].tagname == "inline")


def doctree_update_links(app, doctree, fromdocname):
    """
    Function hook to replace references.
    """
    attrs = ("refuri", "refid")
    link_subs = app.env.config.re_reference
    std_domain = app.env.domains["std"]
    for link in doctree.findall(is_ref_inline):
        for key, repl in link_subs.items():
            if key.startswith("re-") and "=" in key:
                attr, pat = make_pattern(key)
                if attr in link.attributes and pat.match(link.attributes[attr]):
                    original = link.attributes.pop(attr)
                    target = pat.sub(repl, original).lower()
                    if ".md" in target or ".rst" in target:
                        # refid has no reftype
                        ref = std_domain.resolve_xref(
                            app.env,
                            fromdocname,
                            app.builder,
                            link.attributes.get("reftype", "ref"),
                            target,
                            addnodes.pending_xref(
                                "",
                                refdomain="std",
                                reftype=link.attributes.get("reftype", "ref"),
                                reftarget=target,
                                refexplicit=True,
                            ),
                            link.children[0],
                        )
                        if ref is None:
                            logger.warning(
                                "Could not find target to replace %s (%s) with %s.",
                                original, key[3:], target,
                                location=link, type="re_references.invalid_link"
                            )
                            link.attributes[attr] = original
                            continue
                        attribs = ref.attributes
                        _attr = ([None] + [a for a in attrs if a in attribs]).pop()
                        if _attr is None:
                            msg = f"Neither {' or '.join(attrs)} in reference."
                            raise RuntimeError(msg)
                        target = ref[_attr]
                    elif target[0] == "#" or target.startswith(f"/{fromdocname}#"):
                        _attr = "refid"
                    else:
                        _attr = "refuri"
                    link.attributes[_attr] = target
                    logger.info(f"replacing {original} with {target} ({_attr}).")


def missing_references(app, env, node, contnode):
    if node.attributes.get("reftype", "") in ("ref", "myst"):
        substitutions = env.config.re_reference
        std_domain = env.domains["std"]

        for key, repl in substitutions.items():
            if key.startswith("re-") and "=" in key:
                attr, pat = make_pattern(key)
                if attr in node.attributes and pat.match(node.attributes[attr]):
                    node["reftarget"] = pat.sub(repl, node.attributes[attr])
                    ref = std_domain.resolve_xref(
                        env,
                        node["refdoc"],  # fromdocname
                        app.builder,
                        node["reftype"],
                        node["reftarget"],
                        node,
                        contnode,
                    )

                    if ref is None:
                        raise RuntimeError(
                            f"Could not find reference to {node['reftarget']} in "
                            f"{node['refdoc']}!"
                        )
                    attrs = ("reftarget", "refuri", "refuri")
                    target = next((a, ref[a]) for a in attrs if a in ref.attributes)
                    print(f"replacing {node.attributes[attr]} with {'='.join(target)}")
                    return ref
        # std_domain.resolve_xref


def setup(app):
    app.pdb = True
    app.add_config_value("re_reference", {}, "env", dict)
    app.connect("missing-reference", missing_references)
    app.connect("doctree-resolved", doctree_update_links)
