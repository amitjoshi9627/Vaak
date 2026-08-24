from collections.abc import Callable
from typing import Generic, TypeVar

from vaak.core.exceptions import RegistryError

T = TypeVar("T")


class Registry(Generic[T]):
    """Generic name-to-object registry."""

    def __init__(self, name: str) -> None:
        self._name = name
        self._items: dict[str, T] = {}

    @property
    def name(self) -> str:
        """Return the registry name."""

        return self._name

    def register(
        self,
        name: str,
        item: T | None = None,
    ) -> T | Callable[[T], T]:
        """
        Register an item.

        Supports both:

            registry.register("wavlm", encoder)

        and:

            @registry.register("wavlm")
            class WavLMEncoder:
                ...
        """

        if item is not None:
            self._register(name, item)
            return item

        def decorator(obj: T) -> T:
            self._register(name, obj)
            return obj

        return decorator

    def get(self, name: str) -> T:
        """Retrieve an item by name."""

        try:
            return self._items[name]
        except KeyError as exc:
            available = ", ".join(sorted(self._items)) or "<empty>"

            raise RegistryError(
                f"'{name}' not found in {self._name} registry. Available: {available}"
            ) from exc

    def contains(self, name: str) -> bool:
        """Check whether an item exists."""

        return name in self._items

    def names(self) -> tuple[str, ...]:
        """Return registered names."""

        return tuple(sorted(self._items))

    def _register(self, name: str, item: T) -> None:
        if not name:
            raise RegistryError(
                f"Cannot register an item with an empty name in {self._name} registry."
            )

        if name in self._items:
            raise RegistryError(
                f"'{name}' is already registered in {self._name} registry."
            )

        self._items[name] = item
