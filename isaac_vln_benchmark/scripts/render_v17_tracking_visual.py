#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import urllib.request
import zlib
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def post(endpoint: str, body: dict) -> dict:
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Render Grounded-SAM tracking evidence overlays.")
    parser.add_argument("--frame", action="append", required=True, help="label=path")
    parser.add_argument("--target", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--endpoint", default="http://10.100.100.128:8097/detect_segment")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    index = []
    for spec in args.frame:
        label, raw_path = spec.split("=", 1)
        path = Path(raw_path)
        image = Image.open(path).convert("RGB")
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        response = post(
            args.endpoint,
            {
                "image_base64": encoded,
                "image_format": path.suffix.lstrip(".") or "ppm",
                "target_query": args.target,
                "box_threshold": 0.30,
                "text_threshold": 0.25,
                "max_detections": 4,
            },
        )
        detections = response.get("detections") if isinstance(response.get("detections"), list) else []
        selected = next(
            (
                row
                for row in detections
                if isinstance(row, dict)
                and bool((row.get("visual_attribute_evidence") or {}).get("exists", True))
            ),
            None,
        )
        mask_image = Image.new("L", image.size, 0)
        overlay = image.copy()
        draw = ImageDraw.Draw(overlay, "RGBA")
        summary = {"label": label, "target": args.target, "detected": selected is not None}
        if selected is not None:
            mask_raw = zlib.decompress(base64.b64decode(selected["mask_zlib_base64"]))
            mask_image = Image.frombytes("L", image.size, bytes(255 if value else 0 for value in mask_raw))
            tint = Image.new("RGBA", image.size, (0, 0, 0, 0))
            tint.paste((0, 220, 120, 105), mask=mask_image)
            overlay = Image.alpha_composite(overlay.convert("RGBA"), tint).convert("RGB")
            draw = ImageDraw.Draw(overlay, "RGBA")
            x0, y0, x1, y1 = selected.get("bbox_xyxy_norm", [0, 0, 0, 0])
            box = [x0 * image.width, y0 * image.height, x1 * image.width, y1 * image.height]
            draw.rectangle(box, outline=(0, 255, 140, 255), width=3)
            evidence = selected.get("visual_attribute_evidence") or {}
            summary.update(
                {
                    "score": selected.get("score"),
                    "candidate_source": selected.get("candidate_source", "groundingdino_sam2_image"),
                    "bbox_xyxy_norm": selected.get("bbox_xyxy_norm"),
                    "visual_attribute_evidence": evidence,
                    "mask_pixel_fraction": sum(mask_raw) / max(1, len(mask_raw)),
                }
            )
        text = (
            f"{label} | target={args.target} | detected={summary['detected']}"
            + (f" | score={summary.get('score')}" if summary.get("score") is not None else "")
        )
        draw.rectangle([0, 0, image.width, 28], fill=(0, 0, 0, 190))
        draw.text((8, 7), text, fill=(255, 255, 255, 255), font=ImageFont.load_default())
        image.save(output / f"{label}_rgb.png")
        mask_image.save(output / f"{label}_mask.png")
        overlay.save(output / f"{label}_overlay.png")
        (output / f"{label}_response.json").write_text(json.dumps(response, indent=2) + "\n", encoding="utf-8")
        index.append(summary)
    (output / "overlay_state.json").write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(index, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
