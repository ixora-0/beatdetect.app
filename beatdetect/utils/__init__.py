from typing import Dict, List, Union

from .iterate_beat_files import iterate_beat_files

# https://github.com/python/typing/issues/182
JSON = Union[str, int, float, bool, None, Dict[str, "JSON"], List["JSON"]]


__all__ = [
    "iterate_beat_files",
]
