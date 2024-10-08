import json
import os
from pathlib import Path
from typing import Any, Optional
from urllib.parse import ParseResult, unquote, urljoin, urlparse
from urllib.request import url2pathname

from ..formats import CoreFormat, format_from_str
from ..pointcloud.pcl import GlTFPointCloud
from ..project import ProjectResource
from ..types import VersionInfo
from ..versions import Format, get_compatible_type


def join_uris(uri: str, base_uri: Optional[str]) -> ParseResult:
    """Resolve a URI relative to an absolute base URI if the input URI is a relative URI
    reference, otherwise return the URI unmodified.
    The return value is wrapped as a urllib.parse.ParseResult.
    """
    if base_uri is not None:
        uri = urljoin(base_uri + "/", uri)

    url = urlparse(uri)
    if url.hostname is not None and url.hostname != "localhost":
        raise ValueError(
            "Only relative URI references or absolute URIs"
            " referring to the localhost are supported"
        )

    return url


def url_to_path(url: ParseResult) -> Path:
    if url.scheme == "file" or url.scheme == "":
        return Path(url2pathname(url.path))

    raise RuntimeError("A non-file URIs is not accepted")


def _load_from_json(uri: Path) -> Any:
    with open(str(uri)) as f:
        try:
            d = json.load(f)
        except json.decoder.JSONDecodeError as e:
            raise RuntimeError("Error parsing JSON resource %s: %s" % (uri, e)) from e

    try:
        format = format_from_str(d["format"])
        version = VersionInfo.parse(d["version"])
    except KeyError:
        raise RuntimeError("Input file is not a valid OPF JSON resource")

    cls = get_compatible_type(format, version)
    if cls is None:
        raise UnsupportedVersion(format, version)

    try:
        object = cls.from_dict(d)
        if format == CoreFormat.PROJECT:
            # The uri is converted to absolute based on the cwd now because we
            # have no gurantee it won't be changed later and that was the path
            # that was used to successfully
            # open the file above.
            object.base_uri = uri.resolve().parent.as_uri()

        return object
    except Exception as e:
        raise RuntimeError(f"Error decoding JSON resource {format}, {version}") from e


def _ensure_uri(uri: str | os.PathLike) -> str:
    path = Path(uri)  # This doesn't throw if given something like "file:///foo"
    if path.is_absolute():
        return path.as_uri()
    else:
        return str(path)


def _test_json_resource(
    resource: str | ProjectResource | os.PathLike, base_uri: str, _
) -> tuple[bool, Optional[list[Any]]]:
    if isinstance(resource, ProjectResource):
        path = url_to_path(join_uris(resource.uri, base_uri))
    else:
        path = url_to_path(join_uris(_ensure_uri(resource), base_uri))

    if path.suffix == ".json" or path.suffix == ".opf":
        return (True, [path])
    return (False, None)


def _test_gltf_model_resource(
    resource: str | ProjectResource | os.PathLike, base_uri: str, _
) -> tuple[bool, Optional[list[Any]]]:
    if isinstance(resource, ProjectResource):
        if resource.format == CoreFormat.GLTF_MODEL:
            path = url_to_path(join_uris(resource.uri, base_uri))
        else:
            return (False, None)
    else:
        path = url_to_path(join_uris(_ensure_uri(resource), base_uri))

    if path.suffix == ".gltf":
        return (True, [path])
    return (False, None)


def _test_gltf_binary_resource(
    resource: str | ProjectResource | os.PathLike, base_uri: str, _
) -> tuple[bool, Optional[list[Any]]]:
    if (
        isinstance(resource, ProjectResource)
        and resource.format == CoreFormat.GLTF_BUFFER
    ):
        return (True, [])
    return (False, None)


def _test_features_matches_resource(
    resource: str | ProjectResource | os.PathLike, base_uri: str, _
) -> tuple[bool, Optional[list[Any]]]:
    if isinstance(resource, ProjectResource) and (
        resource.format == CoreFormat.FEATURES
        or resource.format == CoreFormat.MATCHES
        or resource.format == CoreFormat.ORIGINAL_MATCHES
    ):
        return (True, [])
    return (False, None)


loaders = [
    (_test_json_resource, _load_from_json),
    (_test_gltf_model_resource, GlTFPointCloud.open),
    # This is used just for skipping glTF binary buffers in the project resolver
    (_test_gltf_binary_resource, lambda: None),
    # This is used just for skipping feature and matches binary buffers in the project resolver
    (_test_features_matches_resource, lambda: None),
]
"""
A resource loader is a tuple of a test function and a loading function.
The test function must accepts a resource URI, a base URI and a list ProjectResource and returns
a tuple with a boolean, which indicates if the resource is loadable, and the list of parameters
that must be passed to the loading function which is derived from the given resources.
"""


class UnsupportedResource(RuntimeError):
    def __init__(self, uri=None):
        self.uri = uri


class UnsupportedVersion(RuntimeError):
    def __init__(self, _format: Format, version: VersionInfo):
        self._format = _format
        self.version = version
        self.message = f"Unsupported resource format and version: {_format}, {version}"

    def __str__(self):
        return self.message


def load(
    resource: str | ProjectResource | os.PathLike,
    base_uri: Optional[str] = None,
    additional_resources: Optional[list[ProjectResource]] = None,
) -> Any:
    """Loads a resource from a URI
    :param resource: a resource to be loaded. It must be a ProjectResource, a string with a URI or path, or
        a os.PathLike object
    :param base_uri: Base URI to use to resolve relative URI references
    :param additional_resouces: Additional resources for resources that require multiple
        not referenced by the main resource file.
    :return: The loaded resource or None if the input URI is an auxiliary resource belonging to
       some other primary resource
    """
    for test, loader in loaders:
        accepted, params = test(resource, base_uri, additional_resources)
        if accepted:
            return loader(*params)

    raise UnsupportedResource(resource)
