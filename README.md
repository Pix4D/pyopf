## Python Open Photogrammetry Format (OPF)

This repository provides a Python package for reading, writing and manipulating projects in the OPF format.
For more information about what OPF is and its full specification, please refer to https://www.github.com/Pix4D/opf-spec

### Installation

The tool can be installed using `pip` with the following command:

```shell
pip install pyopf
```

This command installs the `pyopf` package and tools.


### Structure of the PyOPF repository

The `pyopf` library can be found under `src/pyopf`. The library implements easy parsing and writing of OPF projects in Python.

Below is a small example, printing the calibrated position and orientation of a camera, knowing its ID.

```python
from pyopf.io import load

from pyopf.resolve import resolve
from pyopf.uid64 import Uid64

# Path to the example project file.
project_path = "spec/examples/project.json"

# We are going to search for the calibrated position of the camera with this ID
camera_id = Uid64(hex = "0x57282923")

# Load the json data and resolve the project, i.e. load the project items as named attributes.
project = load(project_path)
project = resolve(project)

# Many objects are optional in OPF. If they are missing, they are set to None.
if project.calibration is None:
    print("No calibration data.")
    exit(1)

# Filter the list of calibrated cameras to find the one with the ID we are looking for.
calibrated_camera = [camera for camera in project.calibration.calibrated_cameras.cameras if camera.id == camera_id]

# Print the pose of the camera.
print("The camera {} is calibrated at:".format(camera_id), calibrated_camera[0].position)
print("with orientation", calibrated_camera[0].orientation_deg)
```

The custom attributes are stored per node in the `custom_attributes` dictionary. This dictionary might be `None` if
the `Node` has no associated custom attributes. Below is an example of setting a custom attribute.

```python
import numpy as np
from pathlib import Path
from pyopf.pointcloud import GlTFPointCloud

pcl = GlTFPointCloud.open(Path('dense_pcl/dense_pcl.gltf'))

# Generate a new point attribute as a random vector of 0s and 1s
# The attribute must have one scalar per point
new_attribute = np.random.randint(0, 2, size=len(pcl.nodes[0]))

# The attribute must have the shape (number_of_points, 1)
new_attribute = new_attribute.reshape((-1, 1))
# Supported types for custom attributes are np.float32, np.uint32, np.uint16, np.uint8
new_attribute = new_attribute.astype(np.uint32)

# Set the new attribute as a custom attribute for the node
# By default, nodes might be missing custom attributes, so the dictionary might have to be created
if pcl.nodes[0].custom_attributes is not None:
    pcl.nodes[0].custom_attributes['point_class'] = new_attribute
else:
    pcl.nodes[0].custom_attributes = {'point_class': new_attribute}

pcl.write(Path('out/out.gltf'))
```

### OPF Tools

We provide a few tools as command line scripts to help manipulate OPF projects in different ways.

#### Merging

The main use case for merging projects is to be able to process smaller sections of a project independently.
For the merging to succeed the sub projects must be in the same coordinate reference system. Note that the tool doesn't support merging the content of most OPF extensions, which will then be dropped in the merged project.
Two objects are considered identical if they have the same ID even if they are in different projects. If this assumption is violated, the merging fails. For example, the same camera ID cannot be associated with two different image URIs.
The only exception are the sensors, whose IDs are always regenerated and for which no attempt is made at finding common and equally calibrated sensors.

The point clouds are merged based on their label.

Only core project items support merging:
* camera list
* input cameras
* projected input cameras
* input control points
* projected control points
* calibration (calibrated cameras, calibrated control points, tracks)
* point clouds
* constraints

All extensions are dropped.

The merging tool can be called using

`opf_merge project_1.opf project_2.opf project_3.opf output_directory`


#### Undistorting

A tool to undistort images is provided. The undistorted images will be stored in their original location, but in an `undistort` directory. Only images taken with a perspective camera, for which the sensor has been calibrated will be undistorted.

This tool can be used as

`opf_undistort project.opf`

#### Cropping

We call "cropping" the operation of preserving only the region of interest of the project (as defined by the Region of
Interest OPF extension).
The project to be cropped *MUST* contain an item of type `ext_pix4d_region_of_interest`.

During the cropping process, only the control points and the part of the point clouds which are contained in the ROI are kept.
Cameras which do not see any remaining points from the point clouds are discarded.
Also, cropping uncalibrated projects is not supported.

The following project items are updated during cropping:
* Point Clouds (including tracks)
* Cameras (input, projected, calibrated, camera list)
* GCPs

The rest of the project items are simply copied.

The cropping tool can be called using

`opf_crop project_to_crop.opf output_directory`

#### Convert to NeRF

This tool converts OPF projects to NeRF. NeRF consists of transforms file(s), which contain information about distortion, intrinsic and extrinsinc parameters of cameras. Usually it is split in `transforms_train.json` and `transforms_test.json` files, but can sometimes also have only the train one. The split can be controlled with the parameter `--train-frac`, for example `--train-frac 0.7` will randomly assign 70% of images for training, and the remaining 30% for testing. If this parameter is unspecified or set to 1.0, only the `transforms_train.json` will be generated. Sometimes an additional `transforms_val.json` is required. It is to evaluate from new points of view, but the generation of new point of views is not managed by this tool, so it can just be a copy of `transforms_test.json` renamed.

The tool can also convert input images to other image formats using `--out-img-format`. An optional output directory can be given with `--out-img-dir`, otherwise the images are written to the same directory as the input ones. If `--out-img-dir` is used without `--out-img-format`, images will be copied. When copying or converting an image, the input directory layout is preserved.

When `--out-img-dir` is used, the tree structure of where input images are stored will be copied to the output image directory. In other words, if all images are stored in the same directory, the folder specified by `--out-img-dir` will only contain the images. If images are stored in different folders/subfolders, the `--out-img-dir` folder will contain the same folders/subfolders starting from the first common folder.

Only calibrated projects with only perspective cameras are supported. Remote files are not supported.

##### Examples

Different NeRFs require different parameter settings, by default all values are set to work with Instant-NeRF, so it can be used as:

`opf2nerf project.opf --output-extension`

DirectVoxGo only works with PNG image files, and contrary to Instant-NeRF it doesn't flip cameras orientation with respect to OPF. Thus it can be used as:

`opf2nerf project.opf --out-img-format png --out-img-dir ./images --no-camera-flip`

## License and citation

If you use this work in your research or projects, we kindly request that you cite it as follows:

The Open Photogrammetry Format Specification, Grégoire Krähenbühl, Klaus Schneider-Zapp, Bastien Dalla Piazza, Juan Hernando, Juan Palacios, Massimiliano Bellomo, Mohamed-Ghaïth Kaabi, Christoph Strecha, Pix4D, 2023, retrived from https://pix4d.github.io/opf-spec/

Copyright (c) 2023 Pix4D SA

All scripts and/or code contained in this repository are licensed under Apache License 2.0.

Third party documents or tools that are used or referred to in this specification are licensed under their own terms by their respective copyright owners.
