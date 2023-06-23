import argparse
import os
from pathlib import Path

# This module and functions must be renamed when adding a new version.
# The implementation needs to be reviewed as well to not use the current API objects,
# but only legacy ones.
from opf_tools.convert.v1_0_draft6_v1_0_draft7 import (
    v1_0_draft6_to_v1_0_draft7,
    v1_0_draft7_to_v1_0_draft6,
)
from opf_tools.convert.v1_0_draft7_v1_0_draft8 import (
    v1_0_draft7_to_v1_0_draft8,
    v1_0_draft8_to_v1_0_draft7,
)
from opf_tools.convert.v1_0_draft8_v1_0_draft9 import (
    v1_0_draft8_to_v1_0_draft9,
    v1_0_draft9_to_v1_0_draft8,
)
from opf_tools.convert.v1_0_draft9_latest import (
    latest_to_v1_0_draft9,
    v1_0_draft9_to_latest,
)
from pyopf.io import load, save
from pyopf.project import ProjectObjects
from pyopf.resolve import resolve
from pyopf.types import OpfObject, VersionInfo
from pyopf.versions import FormatVersion

_version_history = [
    VersionInfo(1, 0, "draft6"),
    VersionInfo(1, 0, "draft7"),
    VersionInfo(1, 0, "draft8"),
    VersionInfo(1, 0, "draft9"),
    VersionInfo(1, 0),
]


_upgraders = [
    v1_0_draft6_to_v1_0_draft7,
    v1_0_draft7_to_v1_0_draft8,
    v1_0_draft8_to_v1_0_draft9,
    v1_0_draft9_to_latest,
    None,
]


_downgraders = [
    None,
    v1_0_draft7_to_v1_0_draft6,
    v1_0_draft8_to_v1_0_draft7,
    v1_0_draft9_to_v1_0_draft8,
    latest_to_v1_0_draft9,
]


def convert(
    project: ProjectObjects, target_version: VersionInfo, base_dir: Path
) -> ProjectObjects:

    try:
        from_version_index = _version_history.index(project.metadata.version)
    except ValueError:
        raise ValueError(f"Unsuppported OPF project version {project.metadata.version}")

    try:
        to_version_index = _version_history.index(target_version)
    except ValueError:
        raise ValueError(f"Unsuppported OPF project version: {target_version}")

    while from_version_index != to_version_index:

        if from_version_index > to_version_index:
            project = _downgraders[from_version_index](project, base_dir)
            from_version_index -= 1
        else:
            project = _upgraders[from_version_index](project, base_dir)
            from_version_index += 1

    return project


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OPF project conversion tool")
    parser.add_argument(
        "input",
        metavar="project.opf",
        type=str,
        help="A list of OPF project files",
    )
    parser.add_argument(
        "outdir", type=str, help="Output directory for the converted project"
    )
    parser.add_argument(
        "--version",
        "-v",
        metavar="version string",
        default="latest",
        type=str,
        help="The desired OPF output project version",
    )
    parser.add_argument(
        "-f",
        "--force",
        dest="force",
        action="store_true",
        default=False,
        help="Do not ask for confirmation when overwriting output files",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    output_dir = args.outdir
    if os.path.exists(output_dir):
        if len(os.listdir(output_dir)) != 0 and not args.force:
            while True:
                try:
                    answer = input(
                        "The output directory is not empty do you want to procced [yN]? "
                    )
                except KeyboardInterrupt:
                    print()
                    exit(-1)
                except EOFError:
                    answer = ""
                if answer == "n" or answer == "N" or answer == "":
                    exit(0)
                if answer == "y" or answer == "Y":
                    break
        needs_make_dir = False
    else:
        needs_make_dir = True

    if args.version == "latest":
        target_version = FormatVersion.PROJECT
    else:
        try:
            target_version = VersionInfo.parse(args.version)
        except ValueError:
            print(f"Invalid version string: {args.version}")
            return -1

    project = load(args.input)

    if project.version == target_version:
        print(f"Project is already at version {target_version}")
        return 0

    project = resolve(project)

    converted_project = convert(project, target_version, base_dir=output_dir)

    if needs_make_dir:
        os.makedirs(output_dir)

    save(converted_project, output_dir + "/project.opf", write_point_cloud_buffers=True)
