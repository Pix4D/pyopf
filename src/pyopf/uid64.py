import os

int_ = int  # The built-in int type
bytes_ = bytes  # The built-in bytes type


class Uid64:
    __slots__ = "int"

    def __init__(
        self,
        int: int_ | None = None,
        hex: str | None = None,
        bytes: bytes_ | None = None,
    ):

        if [hex, bytes, int].count(None) != 2:
            raise TypeError("Only one of int or hex must be given")

        if hex is not None:
            int = int_(hex, 16)

        if bytes is not None:
            if len(bytes) != 8:
                raise ValueError("bytes is not a 8-char string")
            assert isinstance(bytes, bytes_), repr(bytes)
            int = int_.from_bytes(bytes, byteorder="big")

        if int is not None:
            if not 0 <= int < 1 << 64:
                raise ValueError("int is out of range (need a 64-bit value)")
        object.__setattr__(self, "int", int)

    @property
    def bytes(self):
        return self.int.to_bytes(8, "big")

    @property
    def hex(self):
        return self.__str__()

    def __int__(self):
        return self.int

    def __eq__(self, other):
        if isinstance(other, Uid64):
            return self.int == other.int
        if isinstance(other, int_):
            return self.int == other
        return NotImplemented

    def __repr__(self):
        return "%s(%r)" % (self.__class__.__name__, str(self))

    def __setattr__(self, name, value):
        raise TypeError("Uid64 objects are immutable")

    def __str__(self):
        return "0x%016X" % self.int

    def __hash__(self):
        return hash(self.int)

    def __deepcopy__(self, _memo):
        return self


def uid64():
    return Uid64(bytes=os.urandom(8))
