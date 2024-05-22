import argparse
from enum import Enum
from pathlib import Path

import laspy
from pyproj import CRS

from pyopf.crs.crs import Crs
from pyopf.formats import CoreFormat
from pyopf.io import load
from pyopf.pointcloud.pcl import GlTFPointCloud
from pyopf.pointcloud.utils import apply_affine_transform
from pyopf.resolve import resolve

LAS_FILE_EXTENSION = ".las"
LAS_VERSION = laspy.header.Version(1, 4)
PF_WITHOUT_COLOR = laspy.PointFormat(1)
PF_WITH_COLOR = laspy.PointFormat(2)

PRECISION = 10000  # 10^coords_decimals_to_keep


class Dim(str, Enum):
    """Dimension names to use with the PointRecord class from laspy."""

    X = "X"
    Y = "Y"
    Z = "Z"
    R = "red"
    G = "green"
    B = "blue"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a LAS 1.4 pointcloud file from an OPF project.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "opf_path",
        type=str,
        help="[REQUIRED] The path to your project.opf file.",
    )

    parser.add_argument(
        "--out-dir",
        "-o",
        type=str,
        default=str(Path.cwd()),
        help="Output folder for the converted file.",
    )

    return parser.parse_args()


def gltf_to_las(
    gltf: GlTFPointCloud, output_path: Path, crs: Crs | None = None
) -> None:
    """Write a LAS pointcloud from a GlTFPointCloud object and an optional Crs."""

    point_format = (
        PF_WITH_COLOR if gltf.nodes[0].color is not None else PF_WITHOUT_COLOR
    )

    header = laspy.header.LasHeader(version=LAS_VERSION)
    header.offset = [0 for coord in range(3)]
    header.scale = [1 / PRECISION for coord in range(3)]

    if crs is not None:
        crs = CRS.from_wkt(crs.definition)
        try:
            header.add_crs(crs)
        except RuntimeError:
            print(
                "CRS issue, the output LAS file will be written without georeferencing."
            )

    header.point_format = point_format

    las = laspy.create(point_format=point_format, file_version=LAS_VERSION)
    las.header = header
    las.write(str(output_path))

    COORDINATES = [Dim.X, Dim.Y, Dim.Z]
    COLORS = [Dim.R, Dim.G, Dim.B]

    for chunk in gltf.chunk_iterator(yield_indices=False):
        apply_affine_transform(chunk.position, chunk.matrix)
        coords = chunk.position.transpose() * PRECISION
        colors = chunk.color.transpose()

        point_record = laspy.PackedPointRecord.empty(point_format)
        for i in range(3):
            point_record[COORDINATES[i]] = coords[i]
            point_record[COLORS[i]] = colors[i]

        with laspy.open(output_path, "a") as file:
            file.append_points(point_record)


def main():
    args = parse_args()

    opf_path = args.opf_path

    project = load(opf_path)
    resolved_project = resolve(project)
    pointcloud_counter = 0
    for item in project.items:
        for resource in item.resources:
            if resource.format == CoreFormat.GLTF_MODEL:
                gltf_path = Path(opf_path).parent / resource.uri
                output_path = (
                    Path(args.out_dir)
                    / f"{project.name}_{item.name}_{pointcloud_counter}{LAS_FILE_EXTENSION}"
                )
                pointcloud_counter += 1

                print(f"Converting pointcloud {gltf_path} to {output_path}")

                gltf = GlTFPointCloud.open(gltf_path)
                gltf_to_las(
                    gltf, output_path, resolved_project.scene_reference_frame.crs
                )
                print(f"LAS file successfully written: {output_path}\n")


if __name__ == "__main__":
    main()
