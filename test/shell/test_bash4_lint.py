"""
Keep the shipped shell scripts free of constructs bash 3.2 does not have.

macOS ships bash 3.2.57 as ``/bin/bash`` (Apple froze it at the last GPLv2 release), and every
shipped script selects that interpreter through its own ``#!/bin/bash`` shebang, whatever the user's
login shell is. A bash-4-only builtin therefore breaks FastSurfer on macOS while passing every test
we run on Linux, where bash is 5.x.

This scan only reads files, so it is platform independent and runs in the style job on every pull
request. The functional counterpart, which needs a real bash 3.2, lives in test_brun_bash32.py.
"""

import re
from pathlib import Path

import pytest

FASTSURFER_HOME = Path(__file__).parent.parent.parent


def _recon_surf_shell_scripts() -> list[str]:
    """Every bash script in recon_surf, by shebang as well as by suffix.

    Not just *.sh: a shipped executable does not need that name to be run by the pipeline, and one
    without it would silently escape this scan. A .sh file with no shebang counts too, because that
    is how a sourced helper looks, and functions.sh is exactly that. Both rules together leave out
    make_upright (csh) and fs_time (python), which this scan has nothing useful to say about.
    """
    scripts = []
    for path in sorted((FASTSURFER_HOME / "recon_surf").iterdir()):
        if not path.is_file():
            continue
        first_line = path.read_text(errors="replace").split("\n", 1)[0]
        is_bash = "bash" in first_line if first_line.startswith("#!") else path.suffix == ".sh"
        if is_bash:
            scripts.append(str(path.relative_to(FASTSURFER_HOME)))
    return scripts


# Everything that runs under a *macOS* bash: the pipeline entry points, the recon_surf scripts they
# call, the macOS build, and the scripts that run at install time or ship inside the package.
# Deliberately excluded: tools/Docker/entrypoint.sh, tools/export_pip-r.sh and tools/build/fspython
# (all run inside the linux image, the last one activating /venv) and
# CerebNet/datasets/realistic_deformations.sh (a training helper), none of which macOS ever executes.
SHIPPED_SCRIPTS = (
    ["run_fastsurfer.sh", "brun_fastsurfer.sh", "srun_fastsurfer.sh", "long_fastsurfer.sh", "stools.sh"]
    + _recon_surf_shell_scripts()
    + ["tools/build/install_fs_pruned.sh", "tools/build/link_fs.sh", "tools/macos_build/build_release_package.sh"]
    # rendered by the build into the installed package, so they run on the user's machine
    + [
        "tools/macos_build/macos_setup_fastsurfer.sh.template",
        "tools/macos_build/scripts/postinstall.sh.template",
        "tools/macos_build/scripts/preinstall.sh.template",
    ]
)

# A declaration keyword followed by its option list, up to the flag letter the caller appends.
# readonly is included because `readonly -A` is the same bash 4 associative array as `declare -A`.
_DECLARE = r"\b(declare|local|typeset|readonly)\b(\s+-[A-Za-z]+)*\s+-[A-Za-z]*"

# Constructs that bash 3.2 does not have. The value is what to use instead, quoted in the failure.
BASH4_ONLY = {
    r"\bmapfile\b": 'while IFS= read -r line ; do arr+=("$line") ; done < <(...)',
    r"\breadarray\b": 'while IFS= read -r line ; do arr+=("$line") ; done < <(...)',
    # the flag can sit anywhere in the option list and can be bundled with others, so `declare -rA`
    # and `local -r -n ref` have to be caught as well as the plain spelling
    _DECLARE + r"A[A-Za-z]*\b": "parallel indexed arrays, or a delimited string",
    _DECLARE + r"n[A-Za-z]*\b": "pass the value, or eval",
    r"\bcoproc\b": "a named pipe (mkfifo)",
    r"\bwait\s+-n\b": "wait for a specific pid from `jobs -pr`",
    r"\$\{[A-Za-z_][A-Za-z_0-9]*\^\^": "tr '[:lower:]' '[:upper:]'",
    r"\$\{[A-Za-z_][A-Za-z_0-9]*,,": "tr '[:upper:]' '[:lower:]'",
    r"&>>": "'>> file 2>&1'",
}

# GNU-only forms that already cost us a silent failure on macOS and for which the portable spelling
# is simply better, so there is no legitimate reason to write them again:
#   cut --output-delimiter  made image_parameters empty, dropping the t1 path. It is redundant even
#                           on GNU, because cut -f already joins with the input delimiter.
#   expr with \| or \+      is a GNU extension to BRE; BSD expr matched nothing at all, so no
#                           subject parameters parsed. bash's own =~ takes an ERE and needs no expr.
#   sort -V                 is absent from the sort on the macos-15 runners.
#   \s in a regex           is a GNU extension to BRE. BSD sed does not report it, it reads \s as
#                           the letter s, so the subject cleanup silently stripped a trailing "s"
#                           from every line and left trailing whitespace and blank lines in place.
#                           [[:space:]] is POSIX and means the same thing to both.
#
# Deliberately not listed: the *options* of stat, date, readlink and grep. Each has GNU-only ones,
# but a guarded GNU branch is a perfectly good way to use them, as recon_surf/functions.sh does by
# probing `stat --version` first, and a textual check cannot tell that from an unguarded one. The \s
# rule above is different: no probe makes it correct, and [[:space:]] always works.
GNU_ONLY = {
    r"\bcut\b[^|;&]*--output-delimiter": "cut -f already joins with the input delimiter",
    r"\bexpr\b[^|;&]*\\\|": "bash's own =~ with an ERE, which needs no expr",
    r"\bexpr\b[^|;&]*\\\+": "bash's own =~ with an ERE, which needs no expr",
    r"\bsort\b[^|;&]*\s-V\b": "sort -t. -k1,1n -k2,2n -k3,3n",
    # any quantifier, and bare: \s{2} and a lone \s are just as wrong as \s*
    r"\\[sSwW]": "the POSIX class, e.g. [[:space:]]; BSD sed reads \\s as the letter s",
}


def _code_lines(path: Path):
    """Yield (lineno, text) for lines that are not whole-line comments."""
    for number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.lstrip().startswith("#"):
            yield number, line


def _findings(script: str, patterns: dict[str, str]) -> list[str]:
    path = FASTSURFER_HOME / script
    if not path.exists():
        pytest.skip(f"{script} does not exist")
    return [
        f"  {script}:{number}: {line.strip()}\n    use instead: {alternative}"
        for number, line in _code_lines(path)
        for pattern, alternative in patterns.items()
        if re.search(pattern, line)
    ]


# (line, must be flagged). The rules are regexes over source text, so a rule that silently stops
# matching is indistinguishable from a clean repo. These cases pin the spellings down, in both
# directions: the False rows are valid bash 3.2 that must not be reported.
RULE_CASES = [
    ("declare -A colours", True),
    ("declare -rA colours", True),  # bundled with another flag
    ("declare -Ar colours", True),
    ("local -A m", True),
    ("typeset -A m", True),
    ("readonly -A m", True),  # same associative array, different keyword
    ("local -n ref=$1", True),
    ("local -r -n ref=$1", True),  # flag not adjacent to the keyword
    ("mapfile -t arr < <(cmd)", True),
    ("readarray -t arr < file", True),
    ('echo "${name^^}"', True),
    ('echo "${name,,}"', True),
    ("cmd &>> file", True),
    ("wait -n", True),
    ("declare -a list", False),  # indexed array, fine in 3.2
    ("readonly -a list", False),
    ("declare -i count=0", False),
    ("local -r frozen=1", False),
    ('local prev_ifs="$IFS"', False),
    ("export -n VAR", False),  # not a declaration keyword, and valid in 3.2
    ("cmd >> file 2>&1", False),
    ("wait \"${running_jobs[@]}\"", False),
    ('sed "s/[[:space:]]*$//"', False),
    ("cut -d= -f2-1000", False),
]


@pytest.mark.parametrize(("line", "flagged"), RULE_CASES)
def test_rules_match_the_spellings_they_claim_to(line: str, flagged: bool):
    """Each rule fires on the constructs it is for, and stays quiet on valid bash 3.2."""
    rules = {**BASH4_ONLY, **GNU_ONLY}
    hits = [pattern for pattern in rules if re.search(pattern, line)]
    if flagged:
        assert hits, f"no rule flagged {line!r}, which bash 3.2 cannot run"
    else:
        assert not hits, f"{hits} flagged {line!r}, which is valid bash 3.2"


@pytest.mark.parametrize("script", SHIPPED_SCRIPTS)
def test_no_bash4_only_constructs(script: str):
    """No bash 4+ construct in a script that runs under macOS's bash 3.2."""
    findings = _findings(script, BASH4_ONLY)
    assert not findings, "bash 4+ construct in a script that runs under macOS's bash 3.2:\n" + "\n".join(findings)


@pytest.mark.parametrize("script", SHIPPED_SCRIPTS)
def test_no_gnu_only_tool_options(script: str):
    """No GNU-only option to a tool that macOS ships in a BSD flavour."""
    findings = _findings(script, GNU_ONLY)
    assert not findings, "GNU-only option to a tool that is BSD on macOS:\n" + "\n".join(findings)
