"""Registry for constructing named scorer implementations."""

from collections.abc import Callable
from typing import Any


class ScorerRegistry:
    def __init__(self):
        self._factories: dict[str, Callable[..., Any]] = {}

    def register(self, name: str, factory: Callable[..., Any], *, overwrite: bool = False):
        key = str(name).lower()
        if key in self._factories and not overwrite:
            raise KeyError(f"Scorer {name!r} is already registered.")
        self._factories[key] = factory
        return factory

    def build(self, name: str, **kwargs: Any):
        key = str(name).lower()
        try:
            factory = self._factories[key]
        except KeyError as exc:
            available = ", ".join(sorted(self._factories))
            raise KeyError(f"Unknown scorer {name!r}; available scorers: {available}") from exc
        return factory(**kwargs)

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._factories))


SCORER_REGISTRY = ScorerRegistry()


def register_scorer(name: str):
    """Decorator for adding a project-specific scorer factory."""

    def decorator(factory):
        SCORER_REGISTRY.register(name, factory)
        return factory

    return decorator


def get_scorer(name: str, **kwargs: Any):
    return SCORER_REGISTRY.build(name, **kwargs)
