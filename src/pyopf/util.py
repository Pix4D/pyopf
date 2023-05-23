from enum import Enum
from types import UnionType
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, cast

import numpy as np

from pyopf.uid64 import Uid64

from .VersionInfo import VersionInfo

T = TypeVar("T")
EnumT = TypeVar("EnumT", bound=Enum)
IntType = int | np.int64 | np.int32


def from_str(x: Any) -> str:
    assert isinstance(x, str)
    return x


def from_bool(x: Any) -> bool:
    assert isinstance(x, bool)
    return x


def from_list(f: Callable[[Any], T], x: Any) -> List[T]:
    assert isinstance(x, list) or isinstance(x, np.ndarray)
    return [f(y) for y in x]


def from_dict(f: Callable[[Any], T], x: Any) -> Dict[str, T]:
    assert isinstance(x, dict)
    return {k: f(v) for (k, v) in x.items()}


def from_version_info(x: Any) -> VersionInfo:
    assert isinstance(x, VersionInfo)
    return x


def from_none(x: Any) -> Any:
    assert x is None
    return x


def from_uid(x: Any) -> Uid64:
    if isinstance(x, str):
        return Uid64(hex=x)
    if isinstance(x, int):
        return Uid64(int=x)
    if isinstance(x, bytes):
        return Uid64(bytes=x)
    raise ValueError("Unsupported dtype")


def from_union(fs, x):
    for f in fs:
        try:
            return f(x)
        except Exception:
            pass
    assert False


def vector_from_list(
    x: Any, min_size: int = -1, max_size: int = -1, dtype: type | str = "f8"
) -> np.ndarray:
    if max_size != -1 and len(x) > max_size:
        raise ValueError("Invalid array length")
    if max_size != -1 and len(x) < min_size:
        raise ValueError("Invalid array length")

    if (type(dtype) is str and "f" in dtype) or dtype is float:
        return np.array(from_list(from_float, x), dtype=dtype)
    elif (type(dtype) is str and "i" in dtype) or dtype is int:
        return np.array(from_list(from_int, x), dtype=dtype)
    else:
        raise ValueError("Unsupported dtype")


def from_int(x: Any) -> IntType:
    assert isinstance(x, (int, np.int64, np.int32)) and not isinstance(x, bool)  # type: ignore
    return x


def from_float(x: Any) -> float:
    assert isinstance(x, (float, int, np.float32, np.float64)) and not isinstance(  # type: ignore
        x, bool
    )
    return float(x)


def to_float(x: Any) -> float:
    assert isinstance(x, float)
    return x


def to_int(x: Any) -> int:
    assert isinstance(x, (int, np.int64, np.int32))  # type: ignore
    return int(x)


def to_class(
    c: type | UnionType,
    x: "OpfObject | OpfPropertyExtObject",  # noqa: F821 # type: ignore
) -> dict:
    assert isinstance(x, c)
    return x.to_dict()


def to_enum(c: Type[EnumT], x: Any) -> EnumT:
    assert isinstance(x, c)
    return x.value


def from_extensions(x: Any) -> Optional[Dict[str, Dict[str, Any]]]:
    return from_union(
        [lambda x: from_dict(lambda x: from_dict(lambda x: x, x), x), from_none], x
    )
