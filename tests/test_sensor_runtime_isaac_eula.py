from __future__ import annotations

import pytest

from sensor_runtime.isaac_eula import (
    FROZEN_ISAAC_EULA_POLICY,
    FROZEN_ISAAC_RUNTIME_PREFLIGHT,
    ISAAC_EULA_ENVIRONMENT_VARIABLE,
    apply_frozen_isaac_eula_environment,
    build_runtime_preflight,
    require_frozen_runtime_preflight,
)


@pytest.mark.parametrize("inherited", [None, "Y", "YES"])
def test_launcher_normalizes_only_affirmative_repo_conventions(
    inherited: str | None,
) -> None:
    environment = {} if inherited is None else {ISAAC_EULA_ENVIRONMENT_VARIABLE: inherited}
    policy = apply_frozen_isaac_eula_environment(environment)
    assert environment[ISAAC_EULA_ENVIRONMENT_VARIABLE] == "Y"
    assert policy == dict(FROZEN_ISAAC_EULA_POLICY)


@pytest.mark.parametrize("inherited", ["", "N", "NO", "yes", " Y "])
def test_launcher_rejects_negative_or_non_frozen_inherited_values(
    inherited: str,
) -> None:
    with pytest.raises(RuntimeError, match="unsupported inherited"):
        apply_frozen_isaac_eula_environment(
            {ISAAC_EULA_ENVIRONMENT_VARIABLE: inherited}
        )


def test_runtime_preflight_proves_exact_environment_and_devnull() -> None:
    payload = build_runtime_preflight(
        {ISAAC_EULA_ENVIRONMENT_VARIABLE: "Y"},
        stdin_target="/dev/null",
        stdin_is_tty=False,
    )
    assert payload == dict(FROZEN_ISAAC_RUNTIME_PREFLIGHT)
    assert require_frozen_runtime_preflight(payload) == payload


@pytest.mark.parametrize(
    ("environment", "stdin_target", "stdin_is_tty"),
    [
        ({}, "/dev/null", False),
        ({ISAAC_EULA_ENVIRONMENT_VARIABLE: "YES"}, "/dev/null", False),
        ({ISAAC_EULA_ENVIRONMENT_VARIABLE: "Y"}, "pipe:[1]", False),
        ({ISAAC_EULA_ENVIRONMENT_VARIABLE: "Y"}, "/dev/null", True),
    ],
)
def test_runtime_preflight_fails_closed_on_any_policy_drift(
    environment: dict[str, str], stdin_target: str, stdin_is_tty: bool
) -> None:
    with pytest.raises(RuntimeError):
        build_runtime_preflight(
            environment,
            stdin_target=stdin_target,
            stdin_is_tty=stdin_is_tty,
        )
