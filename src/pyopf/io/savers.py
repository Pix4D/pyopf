import json
import os
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote, urlparse

from ..items import CoreItem, ExtensionItem
from ..pointcloud.pcl import GlTFPointCloud
from ..project import (
    Calibration,
    Project,
    ProjectItem,
    ProjectObjects,
    ProjectResource,
)
from ..types import CoreFormat


def to_uri_reference(path, base_path) -> str:
    if base_path:
        return quote(str(path.relative_to(base_path)).replace(os.sep, "/"))
    else:
        return path.as_uri()


def _is_core_json_object(obj: Any):

    # The plan OPF Project type is also treated as bare JSON
    if isinstance(obj, Project):
        return True

    try:
        return (
            obj.format.value.endswith("+json") and obj.format != CoreFormat.GLTF_MODEL
        )
    except AttributeError:
        return False


def _save_to_json(obj: Any, path: Path) -> None:

    with open(path, "w") as out_file:
        json.dump(obj.to_dict(), out_file, indent=4)


def _save_resource_to_json(
    obj: Any, path: Path, base_path: str | Path | None = None, **_
) -> list[ProjectResource]:

    _save_to_json(obj, path)
    return [ProjectResource(format=obj.format, uri=to_uri_reference(path, base_path))]


def _save_point_cloud(
    pcl: GlTFPointCloud,
    output_dir: Path,
    write_point_cloud_buffers: bool = False,
    base_path: str | Path | None = None,
    **_,
) -> list[ProjectResource]:
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    gltf_path = output_dir / "point_cloud.gltf"
    buffer_filepaths = pcl.write(gltf_path, save_buffers=write_point_cloud_buffers)

    resources = [
        ProjectResource(
            format=CoreFormat.GLTF_BUFFER, uri=to_uri_reference(filepath, base_path)
        )
        for filepath in buffer_filepaths
    ]
    resources.append(
        ProjectResource(
            format=CoreFormat.GLTF_MODEL, uri=to_uri_reference(gltf_path, base_path)
        )
    )
    return resources


def _save_project_and_objects(
    project_objs: ProjectObjects,
    path: Path,
    use_item_name_for_resource_uri: bool = False,
    **kwargs,
) -> None:

    base_path = path.parent
    items = []

    def resource_uri_subdir(obj):
        return (
            obj.metadata.name
            if use_item_name_for_resource_uri and obj.metadata.name is not None
            else str(obj.metadata.id)
        )

    def save_subobjects(container, save_function):
        for name, attribute in container.__dict__.items():
            # Skipping private attributes like _metadata
            if name.startswith("_"):
                continue

            if isinstance(attribute, list):
                name_prefix = name[: -len("_objs")]
                if len(attribute) == 1:
                    save_function(name_prefix, attribute[0], base_path)
                else:
                    for i, obj in enumerate(attribute):
                        save_function(f"{name_prefix}_{i}", obj, base_path)
            else:
                if attribute is not None:
                    save_function(name, attribute, base_path)

    def save_object(prefix, obj: CoreItem | ExtensionItem, base_path):

        resources = []

        if isinstance(obj, Calibration):

            subdir = base_path / resource_uri_subdir(obj)
            try:
                os.mkdir(subdir)
            except FileExistsError:
                if not os.path.isdir(subdir):
                    raise RuntimeError(
                        "Fatal error writing object: Path {subdir} exists, but it is not a directory"
                    )

            def save_calibration_subobject(prefix, subobject, base_path):
                output_path = subdir
                # For objects that are not plain JSON resources we will asume the can
                # decide the file names themselves and all they need is the directory
                # where to write.
                if _is_core_json_object(subobject):
                    output_path /= prefix + ".json"

                resources.extend(
                    save(subobject, output_path, base_path=base_path, **kwargs)
                )

            save_subobjects(obj, save_calibration_subobject)

        elif isinstance(obj, GlTFPointCloud):
            resources = save(
                obj, base_path / resource_uri_subdir(obj), base_path=base_path, **kwargs
            )

        elif _is_core_json_object(obj):
            resources = save(
                obj, str(base_path / (prefix + ".json")), base_path=base_path
            )

        assert obj.metadata is not None

        items.append(
            ProjectItem(
                id=obj.metadata.id,
                type=obj.metadata.type,
                name=obj.metadata.name,
                labels=obj.metadata.labels,
                resources=resources,
                sources=obj.metadata.raw_sources(),
            )
        )

    save_subobjects(project_objs, save_object)

    # Saving top level project
    project = Project(
        id=project_objs.metadata.id,
        name=project_objs.metadata.name,
        description=project_objs.metadata.description,
        version=project_objs.metadata.version,
        generator=project_objs.metadata.generator,
        items=items,
    )
    _save_to_json(project, path)


def save(obj: Any, uri: str | Path, **kwargs) -> list[ProjectResource]:
    """Save an OPF object to the given URI.
    :param obj: The object to save. It may be an object directly writable in JSON format, a
        ProjectObjects objet or a GlTFPointCloud
    :param uri: The target destination
    :param kwargs: The following parameters are accepted:
        * write_point_cloud_buffers (bool): If True, the binary buffer files of point
          clouds are also written then saving point clouds.
        * use_item_name_for_resource_uri (bool): Certain items have resources that make a
          bundle (e.g. point clouds). When saving ProjectObjects, these resources are saved
          in a subdirectory relative to the project location. By default the UUID of the item
          is used to name the subdirectory unless this option is set to True, in which case
          the item name will be used.  If the item does not have a name, the UUID is used as
          a fallback
        * base_path (str | Path): An optional parameter to make relative URIs in the
          ProjectResource list returned. This parameter will be ignored is the input object
          is of type ProjectObjects.
    :return: A list of ProjectResources
    """

    if not isinstance(uri, Path):
        uri = Path(unquote(urlparse(uri).path)).absolute()

    for obj_type, saver in savers:
        if isinstance(obj, obj_type):
            return saver(obj, uri, **kwargs)

    if _is_core_json_object(obj):
        return _save_resource_to_json(obj, uri, **kwargs)

    raise RuntimeError("Save is not implemented for this type: %s" % type(obj))


savers = [
    (GlTFPointCloud, _save_point_cloud),
    (ProjectObjects, _save_project_and_objects),
]
"""A object saver is registered a a tuple with made by the object type and a function with signature
   `f(obj: Any, path: Path, **kwargs) -> list[ProjectResource]` as value"""
