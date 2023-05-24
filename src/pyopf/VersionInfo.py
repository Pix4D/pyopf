import functools
import re
from typing import Optional


@functools.total_ordering
class VersionInfo:
    """
    A semver like version class without patch and build numbers
    """

    __slots__ = ("_major", "_minor", "_prerelease")

    #: Regex for a semver version
    _REGEX = re.compile(
        r"""
            ^
            (?P<major>0|[1-9]\d*)
            \.
            (?P<minor>0|[1-9]\d*)
            (?:-(?P<prerelease>[-0-9a-zA-Z-]+))?
            $
        """,
        re.VERBOSE,
    )

    def __init__(self, major: int, minor: int = 0, prerelease: Optional[str] = None):
        if major < 0 or minor < 0:
            raise ValueError("Major and minor version numbers must be positive")

        self._major = major
        self._minor = minor
        self._prerelease = prerelease

    @property
    def major(self):
        """The major part of a version (read-only)."""
        return self._major

    @property
    def minor(self):
        """The minor part of a version (read-only)."""
        return self._minor

    @property
    def prerelease(self):
        """The prerelease part of a version (read-only)."""
        return self._prerelease

    def to_dict(self):
        ret: dict[str, int | str] = {"major": self._major, "minor": self._minor}
        if self._prerelease is not None:
            ret["prerelease"] = self._prerelease
        return ret

    def to_tuple(self):
        if self._prerelease is not None:
            return (self._major, self._minor, self._prerelease)
        else:
            return (self._major, self._minor)

    @classmethod
    def parse(cls, version: str):
        """
        Parse version string to a VersionInfo instance.

        :param version: version string
        :return: a :class:`VersionInfo` instance
        :raises: :class:`ValueError`
        :rtype: :class:`VersionInfo`
        """
        match = cls._REGEX.match(version)
        if match is None:
            raise ValueError("%s is not valid version string" % version)

        version_parts = match.groupdict()

        major = int(version_parts["major"])
        minor = int(version_parts["minor"])
        prerelease = version_parts.get("prerelease", None)

        return cls(major, minor, prerelease)

    def __eq__(self, other):
        return self.to_tuple() == other.to_tuple()

    def __lt__(self, other):
        us = self.to_tuple()
        them = other.to_tuple()
        if len(us) == len(them):
            return us < them
        else:
            return us[0:2] < them[0:2] or (us[0:2] == them[0:2] and len(us) > len(them))

    def __repr__(self):
        s = ", ".join("%s=%r" % (key, val) for key, val in self.to_dict().items())
        return "%s(%s)" % (type(self).__name__, s)

    def __str__(self):
        """str(self)"""
        version = "%d.%d" % (self.major, self.minor)
        if self.prerelease:
            version += "-%s" % self.prerelease
        return version

    def __hash__(self):
        return hash(str(self))
