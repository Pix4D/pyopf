from pathlib import Path
from urllib import parse

import numpy as np
import pygltflib


def gl_to_numpy_type(gl_code: int) -> type:
    """
    Convert the OpenGL codes used by glTF to represent data types into numpy dtypes

    :raises ValueError: if the type is not supported
    """
    match gl_code:
        case 5121:
            return np.uint8
        case 5125:
            return np.uint32
        case 5126:
            return np.float32
        case 5123:
            return np.uint16
        case _:
            raise ValueError(
                "Unsupported or invalid glTF attribute type: code %d" % gl_code
            )


def gl_to_numpy_shape(gl_shape: str) -> int:
    """
    Get the number of elements in a glTF object
    :raises ValueError: if the object type is not supported
    """
    match gl_shape:
        case "SCALAR":
            return 1
        case "VEC2":
            return 2
        case "VEC3":
            return 3
        case "VEC4":
            return 4
        case _:
            raise ValueError(
                "Unsupported or invalid glTF attribute shape: code %d" % gl_shape
            )


def _numpy_to_gl_type(dtype: np.dtype) -> int:
    """
    Convert numpy types into pygltflib codes.
    :raises ValueError if the type is not supported
    """
    match dtype.type:
        case np.float32:
            return pygltflib.FLOAT
        case np.uint32:
            return pygltflib.UNSIGNED_INT
        case np.uint16:
            return pygltflib.UNSIGNED_SHORT
        case np.uint8:
            return pygltflib.UNSIGNED_BYTE
        case _:
            raise ValueError("Unsupported type in glTF " + str(dtype))


def _numpy_to_gl_shape(count: int) -> str:
    """
    Converts the number of elements into an appropriate vector type for pygltflib.
    :raises ValueError: if the count is not supported
    """
    match count:
        case 1:
            return pygltflib.SCALAR
        case 2:
            return pygltflib.VEC2
        case 3:
            return pygltflib.VEC3
        case 4:
            return pygltflib.VEC4
        case _:
            raise ValueError("Unsupported vector type with %s elements" % count)


def merge_arrays(arrays: list[np.ndarray | np.memmap], output_file: Path) -> np.ndarray:
    """Merge multiple 2D numpy arrays in a single memory mapped array, along the first dimension. The second dimension must be the same.

    :param arrays: The list of numpy arrays to merge.
    :param output_file: The path to the memory mapped file to write. If the file is present, it will be overwritten.

    :return: The newly created memory mapped array.

    :raise ValueError: If any of the arrays is not bi-dimensional, if they do not have matching data types
                       or do not agree in the second dimension
    """
    for a in arrays:
        if len(a.shape) != 2:
            raise ValueError("Can only merge bi-dimensional arrays")
        if a.shape[1] != arrays[0].shape[1]:
            raise ValueError("Arrays do not have the same number of columns")
        if a.dtype != arrays[0].dtype:
            raise ValueError("Arrays do not have the same data types")

    total_rows = sum(a.shape[0] for a in arrays)

    newAccessor = np.memmap(
        output_file,
        mode="w+",
        dtype=arrays[0].dtype,
        offset=0,
        shape=(total_rows, arrays[0].shape[1]),
    )

    written_so_far = 0
    for a in arrays:
        newAccessor[written_so_far : written_so_far + a.shape[0], :] = a
        written_so_far += a.shape[0]

    return newAccessor


class Buffer:
    """An abstraction of a glTF buffer whose data is shared by multiple arrays.
    The arrays are merged into a file before writing.
    """

    arrays: list[np.memmap | np.ndarray]

    def __init__(self, buffer_id: int):
        """Create a new buffer entry.
        :param buffer_id: The glTF id of the buffer.
        """
        self.buffer_id = buffer_id
        self.arrays = []

    def add_array(self, data: np.memmap | np.ndarray):
        """Adds a new array of data to the current buffer"""
        self.arrays.append(data)

    def write(self, filepath: Path):
        """Concatenate the data and write to file.
        :param filepath: The file path to write the data to. It is overwritten if present.

        :raise RuntimeError: If the buffer doesn't contain any data.
        :raise ValueError: If the arrays do not match in their second dimension.
        """

        if len(self.arrays) == 0:
            return RuntimeError("There is no data added to the buffer")

        self.arrays = [merge_arrays(self.arrays, filepath)]

    def __len__(self):
        """Returns the total amount of data in the current buffer"""
        return sum([buffer.nbytes for buffer in self.arrays])

    @property
    def number_of_arrays(self):
        """Return the number of arrays used to store the the data of this buffer"""
        return len(self.arrays)

    @property
    def filepath(self):
        """Returns the file path at which the buffer is saved to, in the case where there is only one buffer.`
        :raise RuntimeError: If the object contains multiple arrays or none.
        :raise ValueError: If the buffer is not a memory mapped array.
        """
        if self.number_of_arrays != 1:
            raise RuntimeError("There are none or multiple binary files in this buffer")
        if not hasattr(self.arrays[0], "filename") or self.arrays[0].filename is None:  # type: ignore
            raise ValueError("The buffer is not a memory mapped array")

        return self.arrays[0].filename  # type: ignore


def add_accessor(
    gltf: pygltflib.GLTF2,
    buffers: dict[Path, Buffer],
    data: np.ndarray | np.memmap,
    filepath: Path,
) -> int:
    """
    Adds a new accessor to a GLTF2 object and a corresponding buffer view.
    Assumes there is a one to one correspondence between accessors, buffer views.
    The buffers parameter is also updated with the data to write.
    The GLTF2 object is modified in place.

    :param gltf: The GLTF2 object where to add the new accessor
    :param buffers: A dictionary of buffers, mapping the filepath to a buffer object containing the data associated
                    to that file path. If there is no data for a specific filepath, a new object is created.
    :param data: A numpy array which contains the data for the accessor
                 The format will be inferred from the shape and data type - it is assumed to be row vectors
    :param filepath: The filepath for the binary data. It is assumed to be relative and not contain special characters.

    :return: The id of the new accessor
    """

    accessor_id = len(gltf.accessors)
    buffer_view_id = accessor_id

    if filepath not in buffers:
        buffers[filepath] = Buffer(len(buffers.keys()))

    buffer_id = buffers[filepath].buffer_id

    gltf.accessors.append(
        pygltflib.Accessor(
            bufferView=buffer_view_id,
            type=_numpy_to_gl_shape(data.shape[1]),
            count=data.shape[0],
            componentType=_numpy_to_gl_type(data.dtype),
            min=None,
            max=None,
            byteOffset=None,
        )
    )
    gltf.bufferViews.append(
        pygltflib.BufferView(
            buffer=buffer_id,
            byteOffset=len(buffers[filepath]),
            byteLength=data.nbytes,
            target=pygltflib.ARRAY_BUFFER,
        )
    )

    buffers[filepath].add_array(data)

    return accessor_id


def write_buffers(buffers: dict[Path, Buffer]):
    """Write the buffers to the associated files.
    The file names are taken as the dictionary keys, and must be either relative to the current directory or absolute.

    :raise RuntimeError: If the buffers could not be written.
                         This happens if they do not contain data or their arrays do not match in the second dimension.
    """
    for filepath, buffer in buffers.items():
        buffer.write(filepath)


def add_buffers(gltf: pygltflib.GLTF2, buffers: dict[Path, Buffer], base_path: Path):
    """Register the buffers in the glTF object.
    :param gltf: The GLTF2 object where to add the buffers
    :param buffers: A dictionary mapping a file path to a list of buffers.
    :param base_path: The base path for the relative URI.
    :param save_buffers: If False, the current file paths of the buffers are used. Otherwise, the buffers are assumed
                         to be available at the file paths indicated by the dictionary keys.
    """

    for buffer in buffers.values():
        if buffer.filepath is None:
            raise ValueError("The buffer is not a memory mapped file")
        uri = parse.quote(str(Path(buffer.filepath).relative_to(base_path)))
        gltf.buffers.append(pygltflib.Buffer(byteLength=len(buffer), uri=uri))
