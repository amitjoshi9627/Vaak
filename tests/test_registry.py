import pytest

from vaak.core.exceptions import RegistryError
from vaak.core.registry import Registry


def test_register_and_get() -> None:
    registry: Registry[str] = Registry("test")

    registry.register("example", "hello")

    assert registry.get("example") == "hello"


def test_contains() -> None:
    registry: Registry[str] = Registry("test")

    registry.register("example", "hello")

    assert registry.contains("example")
    assert not registry.contains("missing")


def test_names_are_sorted() -> None:
    registry: Registry[str] = Registry("test")

    registry.register("zebra", "z")
    registry.register("apple", "a")

    assert registry.names() == ("apple", "zebra")


def test_duplicate_registration_fails() -> None:
    registry: Registry[str] = Registry("test")

    registry.register("example", "hello")

    with pytest.raises(RegistryError):
        registry.register("example", "world")


def test_missing_item_fails() -> None:
    registry: Registry[str] = Registry("test")

    with pytest.raises(RegistryError):
        registry.get("missing")
