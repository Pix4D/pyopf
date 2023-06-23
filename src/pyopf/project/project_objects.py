from dataclasses import dataclass, field

from pyopf.project.types import ExtensionProjectItemType

from ..cameras import (
    CalibratedCameras,
    CameraList,
    GpsBias,
    InputCameras,
    ProjectedInputCameras,
)
from ..cps import (
    CalibratedControlPoints,
    Constraints,
    InputControlPoints,
    ProjectedControlPoints,
)
from ..crs import SceneReferenceFrame
from ..formats import ExtensionFormat
from ..items import CoreItem, ExtensionItem
from ..pointcloud.pcl import GlTFPointCloud
from .metadata import ProjectMetadata


@dataclass(eq=False, order=False, kw_only=True)
class Calibration:
    calibrated_cameras_objs: list[CalibratedCameras] = field(default_factory=list)
    calibrated_control_points_objs: list[CalibratedControlPoints] = field(
        default_factory=list
    )
    point_cloud_objs: list[GlTFPointCloud] = field(default_factory=list)
    gps_bias: GpsBias = None

    _metadata: ProjectMetadata = field(default_factory=ProjectMetadata)

    @property
    def calibrated_cameras(self):
        if len(self.calibrated_cameras_objs) != 0:
            return self.calibrated_cameras_objs[0]
        return None

    @property
    def calibrated_control_points(self):
        if len(self.calibrated_control_points_objs) != 0:
            return self.calibrated_control_points_objs[0]
        return None

    @property
    def tracks(self):
        if len(self.point_cloud_objs) != 0:
            return self.point_cloud_objs[0]
        return None

    @property
    def metadata(self):
        return self._metadata

    @metadata.setter
    def metadata(self, metadata):
        self._metadata = metadata


@dataclass(eq=False, order=False, kw_only=True)
class ProjectObjects:

    scene_reference_frame_objs: list[SceneReferenceFrame] = field(default_factory=list)
    camera_list_objs: list[CameraList] = field(default_factory=list)
    input_cameras_objs: list[InputCameras] = field(default_factory=list)
    projected_input_cameras_objs: list[ProjectedInputCameras] = field(
        default_factory=list
    )
    constraints_objs: list[Constraints] = field(default_factory=list)
    input_control_points_objs: list[InputControlPoints] = field(default_factory=list)
    projected_control_points_objs: list[ProjectedControlPoints] = field(
        default_factory=list
    )
    calibration_objs: list[Calibration] = field(default_factory=list)
    point_cloud_objs: list[GlTFPointCloud] = field(default_factory=list)
    extensions: list[ExtensionItem] = field(default_factory=list)

    _metadata: ProjectMetadata = field(default_factory=ProjectMetadata)

    @property
    def metadata(self):
        return self._metadata

    @property
    def scene_reference_frame(self):
        if len(self.scene_reference_frame_objs) != 0:
            return self.scene_reference_frame_objs[0]
        return None

    @property
    def camera_list(self):
        if len(self.camera_list_objs) != 0:
            return self.camera_list_objs[0]
        return None

    @property
    def input_cameras(self):
        if len(self.input_cameras_objs) != 0:
            return self.input_cameras_objs[0]
        return None

    @property
    def projected_input_cameras(self):
        if len(self.projected_input_cameras_objs) != 0:
            return self.projected_input_cameras_objs[0]
        return None

    @property
    def constraints(self):
        if len(self.constraints_objs) != 0:
            return self.constraints_objs[0]
        return None

    @property
    def input_control_points(self):
        if len(self.input_control_points_objs) != 0:
            return self.input_control_points_objs[0]
        return None

    @property
    def projected_control_points(self):
        if len(self.projected_control_points_objs) != 0:
            return self.projected_control_points_objs[0]
        return None

    @property
    def calibration(self):
        if len(self.calibration_objs) != 0:
            return self.calibration_objs[0]
        return None

    @property
    def point_cloud(self):
        if len(self.point_cloud_objs) != 0:
            return self.point_cloud_objs[0]
        return None

    def get_extensions_by_format(
        self, searched_format: ExtensionFormat
    ) -> list[ExtensionProjectItemType]:
        found_extensions = []
        for extension in self.extensions:
            if extension.format == searched_format:
                found_extensions.append(extension)
        return found_extensions
