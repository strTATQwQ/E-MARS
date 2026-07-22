"""Runtime package for vla-nav-panel."""

from typing import Any

from .state import FrontendConfig, FrontendStateStore


def create_app(*args: Any, **kwargs: Any):
    """Import FastAPI only when the optional frontend runtime is requested."""

    from .app import create_app as factory

    return factory(*args, **kwargs)


__all__ = ["FrontendConfig", "FrontendStateStore", "create_app"]
