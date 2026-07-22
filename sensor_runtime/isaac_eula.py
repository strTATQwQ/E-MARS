"""Frozen non-interactive Isaac Sim EULA launch policy and evidence checks."""

from __future__ import annotations

from types import MappingProxyType
from typing import Any, Mapping, MutableMapping


ISAAC_EULA_ENVIRONMENT_VARIABLE = "OMNI_KIT_ACCEPT_EULA"
ISAAC_EULA_EFFECTIVE_VALUE = "Y"
ISAAC_EULA_ACCEPTED_INHERITED_VALUES = frozenset({"Y", "YES"})
ISAAC_RUNTIME_PREFLIGHT_FILENAME = "isaac_runtime_preflight.json"

FROZEN_ISAAC_EULA_POLICY = MappingProxyType(
    {
        "schema_version": 1,
        "status": "FROZEN",
        "environment_variable": ISAAC_EULA_ENVIRONMENT_VARIABLE,
        "effective_value": ISAAC_EULA_EFFECTIVE_VALUE,
        "stdin": "DEVNULL",
        "interactive_prompt_allowed": False,
    }
)

FROZEN_ISAAC_RUNTIME_PREFLIGHT = MappingProxyType(
    {
        "schema_version": 1,
        "status": "PASS",
        "environment_variable": ISAAC_EULA_ENVIRONMENT_VARIABLE,
        "effective_value": ISAAC_EULA_EFFECTIVE_VALUE,
        "stdin_target": "/dev/null",
        "stdin_is_tty": False,
        "before_isaacsim_import": True,
    }
)


def apply_frozen_isaac_eula_environment(
    environment: MutableMapping[str, str],
) -> dict[str, Any]:
    """Normalize the child environment to the repo's frozen Isaac convention.

    An inherited affirmative spelling may be normalized, but an explicit
    negative or unknown value fails closed instead of silently accepting over
    the caller's contrary instruction.
    """

    inherited = environment.get(ISAAC_EULA_ENVIRONMENT_VARIABLE)
    if inherited is not None and inherited not in ISAAC_EULA_ACCEPTED_INHERITED_VALUES:
        raise RuntimeError(
            f"unsupported inherited {ISAAC_EULA_ENVIRONMENT_VARIABLE} value"
        )
    environment[ISAAC_EULA_ENVIRONMENT_VARIABLE] = ISAAC_EULA_EFFECTIVE_VALUE
    return dict(FROZEN_ISAAC_EULA_POLICY)


def build_runtime_preflight(
    environment: Mapping[str, str],
    *,
    stdin_target: str,
    stdin_is_tty: bool,
) -> dict[str, Any]:
    """Prove the effective policy immediately before importing Isaac Sim."""

    if environment.get(ISAAC_EULA_ENVIRONMENT_VARIABLE) != ISAAC_EULA_EFFECTIVE_VALUE:
        raise RuntimeError("Isaac EULA environment differs from the frozen value")
    if stdin_is_tty or stdin_target != "/dev/null":
        raise RuntimeError("Isaac worker stdin is not the frozen non-interactive /dev/null")
    return dict(FROZEN_ISAAC_RUNTIME_PREFLIGHT)


def require_frozen_runtime_preflight(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or value != dict(FROZEN_ISAAC_RUNTIME_PREFLIGHT):
        raise RuntimeError("Isaac runtime preflight differs from the frozen policy")
    return dict(value)
