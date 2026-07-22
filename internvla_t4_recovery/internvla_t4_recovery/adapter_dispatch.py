"""Select the private Lane-B Step3 adapter without changing its public executable."""

from __future__ import annotations

import os


def main(args: list[str] | None = None) -> None:
    if (
        os.environ.get("INTERNNAV_T5_STEP3_LIVE_ADVISOR", "0") == "1"
        or os.environ.get("INTERNNAV_T5_STEP3_DIRECT_HIGH_LEVEL", "0") == "1"
    ):
        from .lane_b_step3_adapter_node import main as selected_main
    else:
        from .adapter_node import main as selected_main
    selected_main(args=args)
