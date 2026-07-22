from isaac_vln_benchmark.reset_on_decision_node import reset_episode_id


def test_transport_reset_episode_id_is_distinct_and_traceable():
    value = reset_episode_id("episode_1", "route_request_abcdef")

    assert value != "episode_1"
    assert value.startswith("episode_1_transport_reset_")
    assert value.endswith("uest_abcdef")
