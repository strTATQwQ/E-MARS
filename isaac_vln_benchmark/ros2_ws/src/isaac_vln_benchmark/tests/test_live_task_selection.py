import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
SPEC = importlib.util.spec_from_file_location(
    "run_live_success_benchmark",
    ROOT / "scripts" / "run_live_success_benchmark.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_select_tasks_can_preserve_materialized_episode_seeds(tmp_path):
    tasks = tmp_path / "tasks.json"
    tasks.write_text(
        '{"tasks": ['
        '{"task_id": "a", "task_type": "semantic_navigation", "seed": 0},'
        '{"task_id": "b", "task_type": "semantic_navigation", "seed": 2}'
        ']}',
        encoding="utf-8",
    )
    config = {
        "benchmark": {
            "tasks_source": str(tasks),
            "seeds": [99],
            "preserve_task_ids": True,
            "preserve_task_seeds": True,
        },
        "tasks": {"semantic_navigation": {"count": 2}},
    }

    selected = MODULE.select_tasks(ROOT, config, None)

    assert [task["seed"] for task in selected] == [0, 2]
