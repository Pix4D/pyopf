from typing import Any, Dict, Optional

from ..types import Extensions, OpfObject
from ..util import (
    from_float,
    from_none,
    from_str,
    from_union,
    to_class,
    to_float,
)


class Crs(OpfObject):
    """Coordinate reference system"""

    """One of:<br>- A [WKT string version
    2](http://docs.opengeospatial.org/is/18-010r7/18-010r7.html).<br>- A string in the format
    `Authority:code+code` where the first code is for a 2D CRS and the second one if for a
    vertical CRS (e.g. `EPSG:4326+5773`). .<br>- A string in the form
    `Authority:code+Auhority:code` where the first code is for a 2D CRS and the second one if
    for a vertical CRS.<br>- A string in the form `Authority:code` where the code is for a 2D
    or 3D CRS.
    """
    definition: str
    geoid_height: Optional[float]
    """Constant geoid height over the underlying ellipsoid in the units of the vertical CRS axis."""

    def __init__(
        self,
        definition: str,
        geoid_height: Optional[float] = None,
    ) -> None:
        super(Crs, self).__init__()
        self.definition = definition
        self.geoid_height = geoid_height

    @staticmethod
    def from_dict(obj: Any) -> "Crs":
        assert isinstance(obj, dict)
        definition = from_str(obj.get("definition"))
        geoid_height = from_union([from_float, from_none], obj.get("geoid_height"))
        result = Crs(definition, geoid_height)
        result._extract_unknown_properties_and_extensions(obj)
        return result

    def to_dict(self) -> dict:
        result = super(Crs, self).to_dict()
        result["definition"] = from_str(self.definition)
        if self.geoid_height is not None:
            result["geoid_height"] = from_union(
                [to_float, from_none], self.geoid_height
            )
        return result

    def __eq__(self, other: "Crs") -> bool:
        # This is a very naïve comparison, but something smarter requires pyproj
        return (
            self.definition == other.definition
            and self.geoid_height == other.geoid_height
        )
