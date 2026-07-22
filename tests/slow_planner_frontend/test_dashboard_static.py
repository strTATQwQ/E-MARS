from __future__ import annotations

from pathlib import Path


PAGE = (
    Path(__file__).resolve().parents[2]
    / "slow_planner_frontend"
    / "static"
    / "index.html"
)


def _page() -> str:
    return PAGE.read_text(encoding="utf-8")


def test_dashboard_uses_quarter_half_quarter_columns() -> None:
    body = _page()
    assert 'grid-template-columns: minmax(0, 1fr) minmax(0, 2fr) minmax(0, 1fr)' in body
    assert 'grid-template-areas: "parameters llm cameras"' in body
    assert 'class="rail parameter-column" aria-label="Parameters and telemetry"' in body
    assert 'class="llm-column" aria-label="LLM workspace"' in body
    assert 'class="rail camera-column" aria-label="Camera workspace"' in body


def test_mission_editor_is_in_the_center_column() -> None:
    body = _page()
    main_start = body.index('<div class="llm-column"')
    mission_start = body.index('<article class="panel mission-panel">')
    cameras_start = body.index('<section class="panel live-panel">')
    main_end = body.index('<aside class="rail camera-column"')
    assert main_start < mission_start < cameras_start < main_end
    assert 'cameraPanels.forEach((node) => cameraColumn.append(node))' in body
    assert 'telemetryPanels.forEach((node) => parameterColumn.append(node))' in body


def test_llm_workspace_prepares_structured_long_context_rationale() -> None:
    body = _page()
    assert 'aria-label="Long context status"' in body
    assert 'id="context-usage"' in body
    assert 'id="context-segments"' in body
    for stage in ("mission_parse", "visual_grounding", "nav_reasoning", "action_plan", "safety_check"):
        assert f'data-reasoning-stage="{stage}"' in body
    assert "Structured summaries only · hidden reasoning remains private" in body


def test_mission_editor_is_browser_local_and_non_authoritative() -> None:
    body = _page()
    assert 'id="mission-input"' in body
    assert 'type="button">Stage locally</button>' in body
    assert "LOCAL DRAFT · CONTROL DISCONNECTED" in body
    assert "not tokenized or dispatched" in body
    assert "<form" not in body.lower()
    assert 'fetch("/api/v1/state"' in body
    assert 'fetch("/api/v1/instruction' not in body


def test_execution_chain_marks_unobserved_control_telemetry() -> None:
    body = _page()
    for stage in ("instruction", "snapshot", "model", "decision", "nav2", "motion"):
        assert f'id="stage-{stage}"' in body
    assert "NOT INSTRUMENTED" in body
    assert "goal acceptance/result telemetry absent" in body
    assert "No raw model generation or hidden chain of thought" in body


def test_rear_camera_uses_the_same_card_geometry_as_forward_views() -> None:
    body = _page()
    assert '<section class="camera-row rear-row"' in body
    assert ".rear-panel { grid-column: auto; }" in body
    assert ".rear-panel .camera-frame" not in body
    assert "aspect-ratio: 16 / 5" not in body
