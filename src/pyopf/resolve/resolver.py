import copy
from pathlib import Path
from typing import Any, List
from urllib.parse import unquote, urljoin, urlparse
from uuid import UUID

from .. import io
from ..pointcloud.pcl import GlTFPointCloud
from ..project import Calibration, Project, ProjectObjects, ProjectSource
from ..project.metadata import Metadata, Sources
from ..project.types import (
    CoreProjectItemType,
    NamedProjectItemType,
    ProjectItemType,
)
from ..types import CoreFormat, Format, format_to_str


def _item_type_to_str(x: ProjectItemType) -> str:
    return x.name if isinstance(x, NamedProjectItemType) else x.value


def _format_to_name(x: Format) -> str:
    if x.startswith("application"):
        prefix_len = len("application/opf-")
        return (x.value[prefix_len:]).split("+")[0].replace("-", "_")
    elif x == CoreFormat.GLTF_MODEL:
        return "point_cloud"
    else:
        raise RuntimeError("Unsupported format " + x)


def _resolve_sources(sources: list[ProjectSource], objects_by_id: dict[UUID, Any]):
    result = Sources()
    for source in sources:
        try:
            obj = objects_by_id[source.id]
            if source.type != obj.metadata.type:
                raise RuntimeError(
                    "Inconsistent project item dependency. "
                    'The source %s was declared as "%s", '
                    'but the item is "%s"'
                    % (source.id, source.type.name, obj.metadata.type.name)
                )
            name = _item_type_to_str(source.type)
            if hasattr(result, name):
                # Only a source of a given type is supported, source will
                # remain unresolved
                return None
            setattr(result, name, obj)

        except KeyError:
            # Not all sources could be resolved
            return None
    return result


def resolve(project: Project, supported_extensions=[]):
    """Take an OPF project and return an object that contains its items
    loaded in named attributes for easier manipulation."""

    result = ProjectObjects()

    objects_by_id = {}

    for item in project.items:
        is_core_item = isinstance(item.type, CoreProjectItemType)
        is_supported_extension = item.type.name in supported_extensions

        if item.type == CoreProjectItemType.CALIBRATION:

            calibration = Calibration()

            calibration.metadata = Metadata.from_item(item)

            for resource in item.resources:

                obj = io.load(resource, project.base_uri, item.resources)
                if obj is None:
                    continue

                name = _format_to_name(resource.format)
                if name == "gps_bias":
                    # Only one GPS bias resource is acceptable
                    if calibration.gps_bias is not None:
                        raise RuntimeError(
                            "A calibration cannnot contain multiple GPS bias resources"
                        )
                    calibration.gps_bias = obj
                else:
                    calibration.__dict__.setdefault(name + "_objs", []).append(obj)

            result.calibration_objs.append(calibration)

        elif len(item.resources) == 1 and (is_core_item or is_supported_extension):
            obj = io.load(item.resources[0].uri, project.base_uri)
            obj.metadata = Metadata.from_item(item)

            if obj.format != item.resources[0].format:
                raise RuntimeError(
                    "Inconsistent resource format detected. The resource %s"
                    ' was declared as "%s", but the target URI contains "%s"'
                    % (item.resources[0].uri, item.resources[0].format, obj.format)
                )

            name = _item_type_to_str(obj.metadata.type)
            objects_by_id[obj.metadata.id] = obj

            if is_core_item:
                result.__dict__[name + "_objs"].append(obj)
            else:
                result.__dict__["extensions"].append(obj)

        elif item.type == CoreProjectItemType.POINT_CLOUD:
            gltf_uri = next(
                resource.uri
                for resource in item.resources
                if resource.format == CoreFormat.GLTF_MODEL
            )
            point_cloud = io.load(gltf_uri, project.base_uri)
            point_cloud.metadata = Metadata.from_item(item)

            result.point_cloud_objs.append(point_cloud)
        else:
            pass

    # Resolving source references
    for obj in objects_by_id.values():

        sources = _resolve_sources(obj.metadata.sources, objects_by_id)
        if sources:
            obj.metadata.sources = sources

    result.metadata.id = project.id
    result.metadata.version = project.version
    result.metadata.name = project.name
    result.metadata.description = project.description

    return result
