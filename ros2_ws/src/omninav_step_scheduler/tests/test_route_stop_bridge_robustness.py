from omninav_step_scheduler.route_stop_primitive_bridge_node import (
    duplicate_request_id,
    source_stamp_out_of_order,
)


def test_duplicate_request_id_is_accepted_once_per_episode_cache():
    seen = set()
    payload = {"request_id": "request_a"}

    assert duplicate_request_id(payload, seen) is False
    assert duplicate_request_id(payload, seen) is True
    seen.clear()
    assert duplicate_request_id(payload, seen) is False


def test_source_stamp_rejects_out_of_order_but_allows_equal_duplicate_time():
    out_of_order, latest = source_stamp_out_of_order({"source_stamp_sec": 10.0}, None)
    assert out_of_order is False
    assert latest == 10.0

    out_of_order, latest = source_stamp_out_of_order({"source_stamp_sec": 10.0}, latest)
    assert out_of_order is False
    assert latest == 10.0

    out_of_order, latest = source_stamp_out_of_order({"source_stamp_sec": 9.9}, latest)
    assert out_of_order is True
    assert latest == 10.0
