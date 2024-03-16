#!/bin/python

# helper script to install environment files

import argparse
import logging
import os.path
import re


logger = logging.getLogger(__name__)


arg_pattern = re.compile('^(\\s*-\\s*)(--[a-zA-Z0-9\\-]+)(\\s+\\S+)?(\\s*(#.*)?)$')
package_pattern = re.compile('^(\\s*-\\s*)([a-zA-Z0-9\\.\\_\\-]+|pip:)(\\s*[<=>~]{1,2}\\s*\\S+)?(\\s*(#.*)?\\s*)$')
dependencies_pattern = re.compile('^\\s*dependencies:\\s*$')


def mode(arg: str) -> str:
    if arg in ["base", "cpu"] or \
            re.match("^cu\\d+$", arg) or \
            re.match("^rocm\\d+\\.\\d+(\\.\\d+)?$"):
        return arg
    else:
        raise argparse.ArgumentTypeError(f"The mode was '{arg}', but should be "
                                         f"'base', 'cpu', 'cuXXX', or 'rocmX.X[.X]', "
                                         f"where X are digits.")


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Filter the yaml files for torch components and return modified files'
    )

    parser.add_argument('-m',
                        '--mode',
                        required=True,
                        type=mode,
                        help="""targeted return:
                        - base: conda environment create without pytorch.
                        - cpu: pytorch install without cuda (only cpu support, only linux)
                        - cu117: standard pytorch install (with cuda 11.7 on linux)
                        - cu118: standard pytorch install (with cuda 11.8 on linux)
                        - rocm5.4.2: rocm pytorch install (with rocm 5.4.2 on linux)
                        """
                        )
    parser.add_argument('-i',
                        '--yaml_env',
                        dest="yaml_in",
                        required=True,
                        help="Path to the input yaml environment"
                        )
    parser.add_argument("-o",
                        dest='yaml_out',
                        default=None,
                        help="Path to the output yaml environment (default: print to stdout)"
                        )
    return parser


def main(args):
    """Function to split a conda env file for pytorch cuda and cpu versions."""

    from operator import xor
    mode = getattr(args, 'mode')
    if mode is None:
        return "ERROR: No mode set."

    yaml_in = getattr(args, 'yaml_in', None)
    if yaml_in is None or not os.path.exists(yaml_in):
        return f"ERROR: yaml environment file {yaml_in} is not valid!"
    with open(yaml_in, "r") as f_yaml:
        lines = f_yaml.readlines()

    out_file = getattr(args, 'yaml_out')
    out_file_pointer = open(out_file, "w") if out_file else None
    # filter yaml file for pip content
    kwargs = {"sep": "", "end": "", "file": out_file_pointer}

    packages_with_device_tag = ["pytorch", "torch", "torchvision", "torchaudio"]
    packages_without_device_tag_but_need_torch = ["torchio"]
    packages_that_only_work_with_cuda = []
    packages_requiring_torch = packages_with_device_tag + packages_without_device_tag_but_need_torch
    all_special_packages = packages_requiring_torch + packages_that_only_work_with_cuda

    in_dep = False
    buffer = ""
    has_package = False
    pip_indent = -1
    has_pip = False

    for line in lines:
        line_stripped = line.lstrip()
        in_dep = in_dep and line_stripped.startswith("-")
        indent_count = len(line) - len(line_stripped)

        # there is something in the buffer, we are changing indents (but not after pip
        # subsection) and there are packages in the buffer, flush the buffer
        logger.debug(f"maybe print buffer: {has_package} {pip_indent} {indent_count}")
        if buffer != "" and has_package and pip_indent in (-1, indent_count):
            has_pip = has_pip or re.search('-\\s*pip', buffer) is not None
            print(buffer, **kwargs)
            buffer = ""
            has_package = False
            pip_indent = -1

        # handle line not part of dependencies
        hits_package = package_pattern.search(line)
        hits_args = arg_pattern.search(line)
        if not in_dep:
            print(line, **kwargs)
            in_dep = dependencies_pattern.search(line) is not None
        # handle lines part of dependencies AND package specs
        elif hits_package is not None:  # no hit
            indent, package_pip, version, comment, _ = hits_package.groups("")

            logger.debug(f"potential package: {mode} - {package_pip} " +
                         f"base {package_pip not in all_special_packages} " +
                         f"not base {package_pip in packages_requiring_torch} " +
                         f"cuda {package_pip in packages_that_only_work_with_cuda}")
            if package_pip == "pip":
                # pip is automatically added in front of the '- pip:' subsection
                pass
            elif package_pip == "pip:":
                buffer = ("" if has_pip else indent + "pip\n") + line
                pip_indent = indent_count
            elif mode == "base" and package_pip not in all_special_packages or \
                    mode != "base" and package_pip in packages_requiring_torch or \
                    mode.startswith("cu") and package_pip in packages_that_only_work_with_cuda:
                if mode != "base" and package_pip in packages_with_device_tag:
                    if "+" in version:
                        version, _ = version.split("+", 1)
                    version += "+" + mode
                buffer += indent + package_pip + version + comment
                has_package = True
        # handle lines part of dependencies AND argument to pip
        elif hits_args is not None:
            # this is an argument line, should only be in pip section
            indent, arg, value, comment, _ = hits_args.groups("")
            if arg in ("--index-url", "--extra-index-url") and "download.pytorch.org" in value:
                value_cpu = re.sub("/whl/[^/]+/?$", f"/whl/{mode}", value)
                buffer += indent + arg + value_cpu + comment
            else:
                buffer += line
        else:
            raise ValueError(f"Invalid line in environment file, could not interpret `{line}`")
        logger.debug("buffer" + buffer)
    if buffer != "" and has_package:
        print(buffer, **kwargs)
    return 0


if __name__ == "__main__":
    import sys
    logging.basicConfig(stream=sys.stderr)
    #logger.setLevel(logging.DEBUG)

    sys.exit(main(make_parser().parse_args()))
