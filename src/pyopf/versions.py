from .VersionInfo import VersionInfo


# To be generated from the schemas
class FormatVersion:
    CALIBRATED_CAMERAS = VersionInfo(1, 0)
    CALIBRATED_CONTROL_POINTS = VersionInfo(1, 0)
    CAMERA_LIST = VersionInfo(1, 0)
    CONSTRAINTS = VersionInfo(1, 0)
    GLTF_OPF_ASSET = VersionInfo(1, 0)
    GPS_BIAS = VersionInfo(1, 0)
    INPUT_CAMERAS = VersionInfo(1, 0)
    INPUT_CONTROL_POINTS = VersionInfo(1, 0)
    PROJECT = VersionInfo(1, 0)
    PROJECTED_CONTROL_POINTS = VersionInfo(1, 0)
    PROJECTED_INPUT_CAMERAS = VersionInfo(1, 0)
    SCENE_REFERENCE_FRAME = VersionInfo(1, 0)


format_and_version_to_type = {}
