from omninav_step_scheduler.internnav_scheduler_node import InternNavSchedulerNode, mode_uses_internnav_recovery


def test_internnav_scheduler_defaults_disabled_until_mode_payload_arrives():
    node = object.__new__(InternNavSchedulerNode)
    node.config = {}
    assert node.internnav_enabled() is False

    node.config = {"internnav": {"enabled": True}}
    assert node.internnav_enabled() is True

    node.config = {"internnav": {"enabled": False}}
    assert node.internnav_enabled() is False


def test_step_internnav_modes_use_internnav_recovery():
    assert mode_uses_internnav_recovery("internnav_only")
    assert mode_uses_internnav_recovery("step_internnav_event")
    assert mode_uses_internnav_recovery("step_internnav_periodic_4_1")
    assert not mode_uses_internnav_recovery("step_omninav_event")
