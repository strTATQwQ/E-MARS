from pathlib import Path
import sys


SCRIPTS = Path(__file__).resolve().parents[4] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from index_mp3d_pe import build_catalog


def test_catalog_selects_only_metric_isaac_scene_root(tmp_path):
    mesh = tmp_path / "scene_a" / "matterport_mesh" / "mesh_a"
    mesh.mkdir(parents=True)
    (mesh / "isaacsim_mesh_a.usd").write_bytes(b"metric")
    (mesh / "isaacsim_mesh_a_non_metric.usd").write_bytes(b"nonmetric")
    (mesh / "fixed.usd").write_bytes(b"fixed")
    (mesh / "isaacsim_mesh_a.obj").write_text(
        "v -1 -2 -3\nv 3 4 5\nf 1 2 1\n", encoding="utf-8"
    )

    catalog = build_catalog(tmp_path)

    assert catalog["scene_count"] == 1
    scene = catalog["scenes"][0]
    assert scene["scene_id"] == "scene_a"
    assert scene["obj_bounds"]["min"] == [-1.0, -2.0, -3.0]
    assert scene["obj_bounds"]["max"] == [3.0, 4.0, 5.0]
