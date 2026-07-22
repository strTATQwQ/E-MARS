from __future__ import annotations

import asyncio
import json

import pytest

pytest.importorskip("fastapi")

from fastapi.routing import APIRoute, APIWebSocketRoute
from starlette.websockets import WebSocketDisconnect

from slow_planner_frontend.app import create_app
from slow_planner_frontend.state import FrontendStateStore


def test_app_exposes_only_frozen_read_apis_plus_root() -> None:
    app = create_app(store=FrontendStateStore())
    http_routes = {
        route.path: set(route.methods or ())
        for route in app.routes
        if isinstance(route, APIRoute)
    }
    websocket_routes = {
        route.path for route in app.routes if isinstance(route, APIWebSocketRoute)
    }
    assert http_routes == {
        "/": {"GET"},
        "/api/v1/health": {"GET"},
        "/api/v1/state": {"GET"},
        "/api/v1/cameras/{view_id}.jpg": {"GET"},
    }
    assert websocket_routes == {"/api/v1/stream"}
    assert not any(
        method in {"POST", "PUT", "PATCH", "DELETE"}
        for methods in http_routes.values()
        for method in methods
    )
    assert app.openapi_url is None
    assert app.docs_url is None


def test_state_and_health_route_payloads_are_sanitized() -> None:
    store = FrontendStateStore()
    store.publish_health({"ready": True, "raw_text": "private"})
    app = create_app(store=store)
    endpoints = {
        route.path: route.endpoint
        for route in app.routes
        if isinstance(route, APIRoute)
    }
    health = json.loads(endpoints["/api/v1/health"]().body)
    state = json.loads(endpoints["/api/v1/state"]().body)
    assert health["lane_id"] == "b"
    assert state["lane_id"] == "b"
    assert "private" not in json.dumps(health)
    assert "private" not in json.dumps(state)


def test_camera_route_rejects_unknown_view() -> None:
    app = create_app(store=FrontendStateStore())
    endpoint = next(
        route.endpoint
        for route in app.routes
        if isinstance(route, APIRoute) and route.path == "/api/v1/cameras/{view_id}.jpg"
    )
    with pytest.raises(Exception) as caught:
        endpoint("../../secret")
    assert getattr(caught.value, "status_code", None) == 404


def test_root_page_keeps_mission_editor_local_without_control_routes() -> None:
    app = create_app(store=FrontendStateStore())
    endpoint = next(
        route.endpoint
        for route in app.routes
        if isinstance(route, APIRoute) and route.path == "/"
    )
    body = endpoint().body.decode("utf-8")
    assert "VLA Nav Panel" in body
    assert "Operator panel" in body
    assert "<form" not in body.lower()
    assert 'id="mission-input"' in body
    assert 'type="button">Stage locally</button>' in body
    assert "LOCAL DRAFT · CONTROL DISCONNECTED" in body
    assert "raw model generation" in body.lower()


def test_root_page_preserves_operator_dashboard_order() -> None:
    app = create_app(store=FrontendStateStore())
    endpoint = next(
        route.endpoint
        for route in app.routes
        if isinstance(route, APIRoute) and route.path == "/"
    )
    body = endpoint().body.decode("utf-8")
    markers = {
        'aria-label="System gauges"',
        'aria-label="Parameters and telemetry"',
        "<h2>Structured decision</h2>",
        'aria-label="LLM workspace"',
        "<h2>LLM Control Workspace</h2>",
        'aria-label="Camera workspace"',
        "<h2>Live robot sensors</h2>",
        'data-camera="front_left"',
        'data-camera="front"',
        'data-camera="front_right"',
        'data-camera="rear"',
        "<h2>Execution chain</h2>",
        "<h2>Go2 / ROS status</h2>",
        "<h2>Runtime</h2>",
    }
    assert all(marker in body for marker in markers)
    assert 'grid-template-columns: minmax(0, 1fr) minmax(0, 2fr) minmax(0, 1fr)' in body
    assert 'grid-template-areas: "parameters llm cameras"' in body


def test_websocket_stream_sends_public_state_without_waiting_for_a_command() -> None:
    app = create_app(store=FrontendStateStore())
    endpoint = next(
        route.endpoint
        for route in app.routes
        if isinstance(route, APIWebSocketRoute) and route.path == "/api/v1/stream"
    )

    class FakeWebSocket:
        accepted = False
        payload = None

        async def accept(self):
            self.accepted = True

        async def send_json(self, payload):
            self.payload = payload
            raise WebSocketDisconnect()

    websocket = FakeWebSocket()
    asyncio.run(endpoint(websocket))
    assert websocket.accepted is True
    assert websocket.payload["lane_id"] == "b"


def test_websocket_rejects_cross_origin_browser_clients() -> None:
    app = create_app(store=FrontendStateStore())
    endpoint = next(
        route.endpoint
        for route in app.routes
        if isinstance(route, APIWebSocketRoute) and route.path == "/api/v1/stream"
    )

    class CrossOriginWebSocket:
        headers = {
            "origin": "https://untrusted.example",
            "host": "127.0.0.1:8300",
        }
        accepted = False
        closed = None

        async def accept(self):
            self.accepted = True

        async def close(self, *, code, reason):
            self.closed = (code, reason)

    websocket = CrossOriginWebSocket()
    asyncio.run(endpoint(websocket))
    assert websocket.accepted is False
    assert websocket.closed == (1008, "cross-origin panel denied")
