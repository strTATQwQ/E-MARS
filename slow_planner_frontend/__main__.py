from __future__ import annotations

import argparse
from pathlib import Path

from .app import create_app
from .state import FrontendConfig


def _load_config(path: Path) -> FrontendConfig:
    import yaml

    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("frontend config must be an object")
    return FrontendConfig.from_mapping(value)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the vla-nav-panel dashboard.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = _load_config(Path(args.config).resolve())

    import uvicorn

    uvicorn.run(
        create_app(config=config),
        host=config.host,
        port=config.port,
        access_log=True,
        server_header=False,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
