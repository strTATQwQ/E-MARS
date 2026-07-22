from __future__ import annotations

import asyncio
from pathlib import Path
from urllib.parse import urlsplit

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, Response

from .state import FilesystemStateSource, FrontendConfig, FrontendStateStore


def _websocket_origin_allowed(websocket: WebSocket) -> bool:
    """Allow non-browser clients or a browser connecting to the serving origin."""

    headers = getattr(websocket, "headers", {})
    origin = str(headers.get("origin") or "")
    if not origin:
        return True
    host = str(headers.get("host") or "")
    parsed = urlsplit(origin)
    return parsed.scheme in {"http", "https"} and parsed.netloc == host


def create_app(
    *,
    config: FrontendConfig | None = None,
    store: FrontendStateStore | None = None,
    source: FilesystemStateSource | None = None,
) -> FastAPI:
    """Create the local operator panel application."""

    active_store = store or FrontendStateStore()
    active_source = source or (
        FilesystemStateSource(config) if config is not None else None
    )
    poll_interval_s = config.poll_interval_s if config is not None else 0.25

    app = FastAPI(
        title="VLA Nav Panel",
        description="Structured state for the four-camera Step3-VL navigation stack.",
        version="1",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.state.frontend_store = active_store
    app.state.frontend_source = active_source

    def refresh() -> None:
        if active_source is not None:
            active_source.refresh(active_store)

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def index() -> HTMLResponse:
        page = Path(__file__).with_name("static").joinpath("index.html")
        return HTMLResponse(
            page.read_text(encoding="utf-8"),
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/api/v1/health", response_class=JSONResponse)
    def health() -> JSONResponse:
        refresh()
        return JSONResponse(
            active_store.health_payload(), headers={"Cache-Control": "no-store"}
        )

    @app.get("/api/v1/state", response_class=JSONResponse)
    def state() -> JSONResponse:
        refresh()
        return JSONResponse(
            active_store.state_payload(), headers={"Cache-Control": "no-store"}
        )

    @app.get("/api/v1/cameras/{view_id}.jpg", response_class=Response)
    def camera(view_id: str) -> Response:
        refresh()
        frame = active_store.camera(view_id)
        if frame is None or not frame.jpeg:
            raise HTTPException(status_code=404, detail="camera frame unavailable")
        headers = {
            "Cache-Control": "no-store",
            "X-Lane": "b",
            "X-Frame-View": frame.view_id,
        }
        if hasattr(frame, "snapshot_id"):
            headers.update(
                {
                    "X-Snapshot-Id": frame.snapshot_id,
                    "X-Sim-Stamp": f"{frame.sim_stamp_s:.9f}",
                    "X-Extrinsic-SHA256": frame.extrinsic_sha256,
                }
            )
        else:
            headers.update(
                {
                    "X-ROS-Stamp": f"{frame.stamp_s:.9f}",
                    "X-ROS-Topic": frame.source_topic,
                }
            )
        return Response(
            content=frame.jpeg,
            media_type="image/jpeg",
            headers=headers,
        )

    @app.websocket("/api/v1/stream")
    async def stream(websocket: WebSocket) -> None:
        if not _websocket_origin_allowed(websocket):
            await websocket.close(code=1008, reason="cross-origin panel denied")
            return
        await websocket.accept()
        last_version = -1
        heartbeat_at = 0.0
        try:
            while True:
                refresh()
                payload = active_store.state_payload()
                now = asyncio.get_running_loop().time()
                version = int(payload["version"])
                if version != last_version or now >= heartbeat_at:
                    await websocket.send_json(payload)
                    last_version = version
                    heartbeat_at = now + 2.0
                await asyncio.sleep(poll_interval_s)
        except (WebSocketDisconnect, RuntimeError):
            return

    return app
