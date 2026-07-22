from isaac_vln_benchmark.benchmark_runner import mode_schedule_options, paired_mode_work_items


def test_paired_mode_schedule_keeps_each_task_pair_adjacent_and_randomized():
    tasks = [{"task_id": f"task_{index}"} for index in range(8)]
    modes = ["baseline", "candidate"]

    items = paired_mode_work_items(tasks, modes, randomize_by_task=True, seed=7)

    assert len(items) == 16
    orders = []
    for offset in range(0, len(items), 2):
        pair = items[offset : offset + 2]
        assert pair[0][0] == pair[1][0]
        assert pair[0][2]["task_id"] == pair[1][2]["task_id"]
        assert {pair[0][1], pair[1][1]} == set(modes)
        orders.append(tuple(row[1] for row in pair))
    assert len(set(orders)) == 2


def test_nonrandom_schedule_preserves_mode_major_order():
    tasks = [{"task_id": "a"}, {"task_id": "b"}]
    items = paired_mode_work_items(tasks, ["baseline", "candidate"], randomize_by_task=False, seed=1)
    assert [(index, mode, task["task_id"]) for index, mode, task in items] == [
        (0, "baseline", "a"),
        (1, "baseline", "b"),
        (0, "candidate", "a"),
        (1, "candidate", "b"),
    ]


def test_mode_schedule_reads_nested_live_success_benchmark_config():
    randomize, seed = mode_schedule_options(
        {"benchmark": {"randomize_mode_order_by_task": True, "mode_order_seed": 20260711}},
        42,
    )
    assert randomize is True
    assert seed == 20260711
