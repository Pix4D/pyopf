from typing import Any, Dict, List, Optional
from uuid import UUID

from ..formats import (
    CoreFormat,
    Format,
    NamedFormat,
    format_from_str,
    from_format,
)
from ..types import OpfObject, VersionInfo
from ..util import (
    from_list,
    from_none,
    from_str,
    from_union,
    from_version_info,
    to_class,
)
from ..versions import FormatVersion, format_and_version_to_type
from .types import (
    NamedProjectItemType,
    ProjectItemType,
    from_project_item_type,
    project_item_type_from_str,
)


def _item_type_to_str(x: ProjectItemType) -> str:
    return x.name if isinstance(x, NamedProjectItemType) else x.value


class Generator:
    """The generator of this project"""

    """The name of the generator"""
    name: str
    """The version of the generator`"""
    version: str

    def __init__(self, name: str, version: str) -> None:
        self.name = name
        self.version = version

    @staticmethod
    def from_dict(obj: Any) -> "Generator":
        assert isinstance(obj, dict)
        name = from_str(obj.get("name"))
        version = from_str(obj.get("version"))
        return Generator(name, version)

    def to_dict(self) -> dict:
        result: dict = {}
        result["name"] = from_str(self.name)
        result["version"] = from_str(self.version)
        return result


class ProjectResource(OpfObject):
    """The storage format of this resource."""

    format: Format
    """URI reference of the resource file as specified by
    [RFC2396](https://www.w3.org/2001/03/identification-problem/rfc2396-uri-references.html).
    If the reference is relative, it is relative to the folder containing the present file
    """
    uri: str

    def __init__(
        self,
        format: Format,
        uri: str,
    ) -> None:
        super(ProjectResource, self).__init__()
        self.format = format
        self.uri = uri

    @staticmethod
    def from_dict(obj: Any) -> "ProjectResource":
        assert isinstance(obj, dict)
        format = from_union([from_format, format_from_str], obj.get("format"))
        uri = from_str(obj.get("uri"))
        result = ProjectResource(format, uri)
        result._extract_unknown_properties_and_extensions(obj)
        return result

    def to_dict(self) -> dict:
        result = super(ProjectResource, self).to_dict()
        if isinstance(self.format, NamedFormat):
            result["format"] = self.format.name
        else:
            result["format"] = self.format.value
        result["uri"] = from_str(self.uri)
        return result


class ProjectSource(OpfObject):
    id: UUID
    type: ProjectItemType

    def __init__(
        self,
        id: UUID,
        type: ProjectItemType,
    ) -> None:
        super(ProjectSource, self).__init__()
        self.id = id
        self.type = type

    @staticmethod
    def from_dict(obj: Any) -> "ProjectSource":
        assert isinstance(obj, dict)
        id = UUID(obj.get("id"))
        type = from_union(
            [from_project_item_type, project_item_type_from_str], obj.get("type")
        )
        result = ProjectSource(id, type)
        result._extract_unknown_properties_and_extensions(obj)
        return result

    def to_dict(self) -> dict:
        result = super(ProjectSource, self).to_dict()
        result["id"] = str(self.id)
        result["type"] = _item_type_to_str(self.type)
        return result


class ProjectItem(OpfObject):
    id: UUID
    """The name of this item"""
    name: Optional[str]
    """The resources that constitute this item"""
    resources: List[ProjectResource]
    """The sources of this items, that is the set of items this item depends on"""
    sources: List[ProjectSource]
    """Define the type of data represented by the item."""
    type: ProjectItemType
    """Labels associated to the item"""
    labels: Optional[List[str]]

    def __init__(
        self,
        id: UUID,
        type: ProjectItemType,
        resources: List[ProjectResource],
        sources: List[ProjectSource],
        name: Optional[str] = None,
        labels: Optional[List[str]] = None,
    ) -> None:
        super(ProjectItem, self).__init__()
        self.id = id
        self.name = name
        self.resources = resources
        self.sources = sources
        self.type = type
        self.labels = labels

    @staticmethod
    def from_dict(obj: Any) -> "ProjectItem":
        assert isinstance(obj, dict)
        id = UUID(obj.get("id"))
        name = from_union([from_str, from_none], obj.get("name"))
        resources = from_list(ProjectResource.from_dict, obj.get("resources"))
        sources = from_list(ProjectSource.from_dict, obj.get("sources"))
        type = from_union(
            [from_project_item_type, project_item_type_from_str], obj.get("type")
        )
        labels = from_union(
            [lambda x: from_list(from_str, x), from_none], obj.get("labels")
        )
        result = ProjectItem(id, type, resources, sources, name, labels)
        result._extract_unknown_properties_and_extensions(obj)
        return result

    def to_dict(self) -> dict:
        result = super(ProjectItem, self).to_dict()
        result["id"] = str(self.id)
        if self.name is not None:
            result["name"] = from_union([from_str, from_none], self.name)
        result["resources"] = from_list(
            lambda x: to_class(ProjectResource, x), self.resources
        )
        result["sources"] = from_list(
            lambda x: to_class(ProjectSource, x), self.sources
        )
        if isinstance(self.type, NamedProjectItemType):
            result["type"] = self.type.name
        else:
            result["type"] = self.type.value
        if self.labels is not None:
            result["labels"] = from_union(
                [lambda x: from_list(from_str, x), from_none], self.labels
            )
        return result


class Project(OpfObject):
    """Project Structure"""

    """The description of the project"""
    description: str
    """The generator of this project"""
    generator: Optional[Generator]
    id: UUID
    """The items contained in this project"""
    items: List[ProjectItem]
    """The name of the project"""
    name: str
    """The version of this specification as `MAJOR.MINOR`. Breaking changes are reflected by a
    change in MAJOR version. Can optionally include a pre-release tag `MAJOR.MINOR-tag`.
    Examples: `0.1`, `1.0`, `1.0-draft1`
    """
    version: VersionInfo

    base_uri: Optional[str] = None
    """Base URI to be used to resolve relative URI reference of project resources."""

    format = CoreFormat.PROJECT

    def __init__(
        self,
        id: UUID,
        name: str,
        description: str,
        items: List[ProjectItem],
        version: VersionInfo = FormatVersion.PROJECT,
        generator: Optional[Generator] = None,
    ) -> None:
        super(Project, self).__init__()
        self.description = description
        self.generator = generator
        self.id = id
        self.items = items
        self.name = name
        self.version = version

    @staticmethod
    def from_dict(obj: Any) -> "Project":
        assert isinstance(obj, dict)
        description = from_str(obj.get("description"))
        assert from_str(obj.get("format")) == CoreFormat.PROJECT

        generator = from_union([Generator.from_dict, from_none], obj.get("generator"))
        id = UUID(obj.get("id"))
        items = from_list(ProjectItem.from_dict, obj.get("items"))
        name = from_str(obj.get("name"))
        version = from_union([from_version_info, VersionInfo.parse], obj.get("version"))
        result = Project(
            id,
            name,
            description,
            items,
            version,
            generator,
        )
        result._extract_unknown_properties_and_extensions(obj, ["format"])
        return result

    def to_dict(self) -> dict:
        result = super(Project, self).to_dict()
        result["description"] = from_str(self.description)
        result["format"] = CoreFormat.PROJECT

        if self.generator is not None:
            result["generator"] = from_union(
                [lambda x: to_class(Generator, x), from_none],
                self.generator,
            )
        result["id"] = str(self.id)
        result["items"] = from_list(lambda x: to_class(ProjectItem, x), self.items)
        result["name"] = from_str(self.name)
        result["version"] = str(self.version)
        return result


format_and_version_to_type[(CoreFormat.PROJECT, FormatVersion.PROJECT)] = Project
