# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

### Added
### Changed
### Fixed
### Removed

## 1.4.0

### Added

- Added the functionality for `opf2nerf` to output the `avg_pos` and `scale` when using with `--nerfstudio`.
- Added the original resources to the metadata of resolved project items.

### Fixed

- Version check for compatibility.

## 1.3.1

### Fixed

- `height` in ROI extension was fixed to `thickness` to comply with OPF specification
- Fix bug causing GlTFPointCloud instances to inherit previous instance nodes

## 1.3.0

### Added

- Parameter to opf2nerf to produce Nerfstudio-ready outputs
- Example script to compute the reprojection error of input GCPs in calibrated cameras

## 1.2.0

### Added

- transformation_matrix property to BaseToTranslatedCanonicalCrsTransform
- OPF pointcloud to COLMAP converter
- OPF pointcloud to PLY converter
- OPF pointcloud to LAS converter
- Support for the Pix4D polygonal mesh extension
- Support for the Pix4D input and calibrated ITPs extension

### Changed

- Raise a KeyError exception if a required attribute is missing
- Make pyopf.io.load accept paths as strings or os.PathLike objects
- Fixed handling of pathlib.Path in pyopf.io.save
- Move to poetry as package manager

### Removed

- OPF projects merging tool

## 1.1.1

### Added
### Changed

- Added missing dependencies pillow and tqdm

### Removed

## 1.1.0

### Added

- opf2nerf converter to easily convert OPF projects to the input required by NVIDIA Instant NERFs (https://github.com/NVlabs/instant-ngp)
- Support for the Pix4D plane extension

### Changed

- Improve quality of the image undistortion tool, and correctly handles the cases where the distortion coefficient are all zero.
- Renamed BaseItem to CoreItem

### Removed

## 1.0.0

### Added

- Initial release

### Changed


### Removed
