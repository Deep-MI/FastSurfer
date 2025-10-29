# Copyright 2024 Image Analysis Lab, German Center for Neurodegenerative Diseases (DZNE), Bonn
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import sys
from pathlib import Path
from xml.etree import ElementTree


def make_parser() -> argparse.ArgumentParser:
    """
    Create a command line interface and return command line options.

    Returns
    -------
    options
        Namespace object holding options.
    """
    parser = argparse.ArgumentParser(
        description="Script to edit provided distribution.xml file.",
    )
    parser.add_argument(
        "--file", "-f",
        type=Path,
        dest="file",
        help="path to distribution file to edit",
        required=True,
    )
    parser.add_argument(
        "--title", "-t",
        type=str,
        dest="title",
        help="value for the title of the distribution file",
        required=True
    )
    return parser


def edit_distribution(distribution_file: Path | str, title: str) -> None:
    """
    Take path to distribution file and edit it.

    Parameters
    ----------
    distribution_file : Path, str
        Path to distribution file.
    title : str
        Value for the title tag.
    """
    dist_elementtree = ElementTree.parse(distribution_file)

    root_tag = dist_elementtree.getroot()
    
    title_tag = ElementTree.SubElement(root_tag, "title")
    title_tag.text = title
    
    background_tag = ElementTree.SubElement(root_tag, "background")
    background_tag.attrib["file"] = "fastsurfer.png"
    background_tag.attrib["uti"] = "public.png"
    background_tag.attrib["scaling"] = "proportional"
    background_tag.attrib["alignment"] = "bottomleft"

    background_darkAqua_tag = ElementTree.SubElement(root_tag, "background-darkAqua")
    background_darkAqua_tag.attrib = background_tag.attrib

    license_tag = ElementTree.SubElement(root_tag, "license")
    license_tag.attrib["file"] = "LICENSE.txt"
    license_tag.attrib["mime-type"] = "text/txt"

    conclusion_tag = ElementTree.SubElement(root_tag, "conclusion")
    conclusion_tag.attrib["file"] = "MACOS.md"
    conclusion_tag.attrib["mime-type"] = "text/txt"
    
    ElementTree.indent(dist_elementtree, '    ')

    dist_elementtree.write(distribution_file,  encoding="utf-8", xml_declaration=True)

    
if __name__ == "__main__":
    parser = make_parser()
    args = parser.parse_args()

    print(f"Editing distribution file: {args.file} ...")
    edit_distribution(args.file, args.title)
    sys.exit(0)
