from dataclasses import dataclass, field
from typing import Optional, Union
from uuid import UUID, uuid4

from ..VersionInfo import VersionInfo
from ..versions import FormatVersion
from .project import Generator, ProjectItem, ProjectItemType, ProjectSource


class Sources:
    """Placeholder class for declaring a project object sources as named properties"""

    pass


@dataclass(order=False, kw_only=True)
class Metadata:
    type: ProjectItemType
    id: UUID = field(default_factory=uuid4)
    name: Optional[str] = None
    labels: Optional[list[str]] = None
    sources: Union[list[ProjectSource], Sources] = field(default_factory=list)
    """The object sources. This may contain an object with named attributes
    pointing to the sources or a list of ProjectSources"""

    @staticmethod
    def from_item(item: ProjectItem) -> "Metadata":
        return Metadata(
            id=item.id,
            name=item.name,
            type=item.type,
            labels=item.labels,
            sources=item.sources,
        )

    def raw_sources(self) -> list[ProjectSource]:
        """Undoes source resolution and returns a list of project sources."""
        if type(self.sources) is list:
            return self.sources

        return [
            ProjectSource(id=obj.metadata.id, type=obj.metadata.type)
            for obj in self.sources.__dict__.values()
        ]


@dataclass(order=False, kw_only=True)
class ProjectMetadata:
    id: UUID = field(default_factory=uuid4)
    name: str = ""
    description: str = ""
    version: VersionInfo = field(default_factory=lambda: FormatVersion.PROJECT)
    generator: Optional[Generator] = None
