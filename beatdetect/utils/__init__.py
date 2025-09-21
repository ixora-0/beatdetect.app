from .set_seed import set_seed

# https://github.com/python/typing/issues/182
JSON = str | int | float | bool | None | dict[str, "JSON"] | list["JSON"]

__all__ = ["set_seed"]
