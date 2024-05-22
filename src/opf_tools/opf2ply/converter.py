import argparse
from pathlib import Path
from tempfile import mkdtemp

import numpy as np
from plyfile import PlyData, PlyElement

from pyopf.formats import CoreFormat
from pyopf.io import load
from pyopf.pointcloud import GlTFPointCloud
from pyopf.pointcloud.utils import apply_affine_transform

PLY_FILE_EXTENSION = ".ply"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export all point clouds from an OPF project as PLY files.",
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


def gltf_to_ply(gltf: GlTFPointCloud, output_path: Path) -> None:
    """Write a PLY pointcloud from a GlTFPointCloud object."""

    COORDS_DTYPE = "f4"
    COLORS_DTYPE = "u1"
    NORMALS_DTYPE = "f4"

    dtype = [(coord, COORDS_DTYPE) for coord in ["x", "y", "z"]]

    total_gltf_len = sum([len(node) for node in gltf.nodes])
    if total_gltf_len:
        node = gltf.nodes[0]

        has_color = node.color is not None
        has_normal = node.normal is not None

        if has_color:
            dtype += [(color, COLORS_DTYPE) for color in ["red", "green", "blue"]]

        if has_normal:
            dtype += [(axis, NORMALS_DTYPE) for axis in ["nx", "ny", "nz"]]

    points = np.memmap(
        mkdtemp() / Path("tempfile"),
        dtype=dtype,
        mode="w+",
        shape=total_gltf_len,
    )

    for chunk, start, end in gltf.chunk_iterator():
        apply_affine_transform(chunk.position, chunk.matrix)

        points[start:end] = list(
            zip(
                *[chunk.position[:, i] for i in range(3)],
                *[chunk.color[:, i] for i in range(3) if has_color],
                *[chunk.normal[:, i] for i in range(3) if has_normal],
            )
        )

    elements = [PlyElement.describe(points, "vertex")]

    PlyData(elements).write(str(output_path))


def main():
    args = parse_args()

    opf_path = args.opf_path

    project = load(opf_path)
    pointcloud_counter = 0
    for item in project.items:
        for resource in item.resources:
            if resource.format == CoreFormat.GLTF_MODEL:
                gltf_path = Path(opf_path).parent / resource.uri
                output_path = (
                    Path(args.out_dir)
                    / f"{project.name}_{item.name}_{pointcloud_counter}{PLY_FILE_EXTENSION}"
                )
                pointcloud_counter += 1

                print(f"Converting pointcloud {gltf_path} to {output_path}")

                gltf = GlTFPointCloud.open(gltf_path)
                gltf_to_ply(gltf, output_path)


if __name__ == "__main__":
    main()
