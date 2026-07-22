from omninav_step_scheduler.internnav_go2_client_node import (
    compose_instruction_text,
    ProgressRecoveryGuard,
    extract_action_code,
    make_agent_config,
    normalize_depth,
    parse_tokens,
    parse_instruction_token_profiles,
    reset_payload,
    select_instruction_tokens,
    serialize_obs,
    strip_local_obs_metadata,
    synthetic_depth,
    synthetic_rgb,
)


def test_extract_action_code_accepts_internnav_shapes():
    assert extract_action_code([{"action": [1], "ideal_flag": True}]) == 1
    assert extract_action_code({"action": [[2]]}) == 2
    assert extract_action_code([[3]]) == 3
    assert extract_action_code([]) == 0


def test_synthetic_observation_arrays_have_expected_shape_and_dtype():
    import pytest

    pytest.importorskip("numpy")
    rgb = synthetic_rgb(3, 256, 256)
    depth = synthetic_depth(256, 256)
    assert rgb.shape == (256, 256, 3)
    assert depth.shape == (256, 256, 1)
    assert str(rgb.dtype) == "uint8"
    assert str(depth.dtype) == "float32"
    encoded = serialize_obs([{"rgb": rgb, "depth": depth}])
    assert len(encoded) > 1000


def test_make_agent_config_contains_cma_model_settings():
    cfg = make_agent_config(
        server_host="10.100.100.128",
        server_port=8087,
        model_name="cma",
        ckpt_path="checkpoints/r2r/fine_tuned/cma_plus",
    )
    assert cfg["model_settings"]["policy_name"] == "CMA_Policy"
    assert cfg["model_settings"]["env_num"] == 1
    assert parse_tokens("1, 2; bad, 3") == [1, 2, 3]


def test_depth_is_normalized_for_cma_observation_space():
    import pytest

    np = pytest.importorskip("numpy")
    depth = np.asarray([[[0.0], [5.0], [12.0]]], dtype=np.float32)
    normalized = normalize_depth(depth, 10.0)
    assert normalized.tolist() == [[[0.0], [0.5], [1.0]]]
    assert normalize_depth(depth, 0.0).tolist() == depth.tolist()


def test_reset_payload_matches_internnav_server_contract():
    assert reset_payload() == {"reset_index": None}
    assert reset_payload([0]) == {"reset_index": [0]}


def test_progress_recovery_overrides_stalled_repeated_turns():
    guard = ProgressRecoveryGuard(
        enabled=True,
        window=3,
        min_yaw_progress_rad=0.05,
        min_xy_progress_m=0.02,
        forward_steps=2,
    )
    assert guard.select_action(2, (0.0, 0.0), 0.0)[0] == 2
    assert guard.select_action(2, (0.0, 0.0), 0.01)[0] == 2
    code, detail = guard.select_action(2, (0.0, 0.0), 0.02)
    assert code == 1
    assert detail["override_active"] is True
    assert detail["override_reason"] == "deadlock_recovery_forward"
    code, detail = guard.select_action(2, (0.0, 0.0), 0.02)
    assert code == 1
    assert detail["override_active"] is True


def test_progress_recovery_keeps_turn_when_pose_is_progressing():
    guard = ProgressRecoveryGuard(
        enabled=True,
        window=3,
        min_yaw_progress_rad=0.05,
        min_xy_progress_m=0.02,
        forward_steps=2,
    )
    assert guard.select_action(2, (0.0, 0.0), 0.0)[0] == 2
    assert guard.select_action(2, (0.0, 0.0), 0.04)[0] == 2
    code, detail = guard.select_action(2, (0.0, 0.0), 0.09)
    assert code == 2
    assert detail["override_active"] is False


def test_instruction_context_selects_profile_tokens():
    text = compose_instruction_text("Go to the red exit sign.", "red exit sign", "stop near the target")
    profiles = parse_instruction_token_profiles({"exit sign": "10, 11, 12"})
    tokens, source = select_instruction_tokens(text, [1, 2, 3], profiles)
    assert tokens == [10, 11, 12]
    assert source == "profile:exit sign"
    fallback, fallback_source = select_instruction_tokens("unknown target", [1, 2, 3], {})
    assert fallback == [1, 2, 3]
    assert fallback_source == "fallback_static_cma_tokens"


def test_strip_local_obs_metadata_removes_context_before_model_send():
    obs = [{"instruction_tokens": [1, 2, 3], "instruction_context": {"subgoal": "exit sign"}}]
    context = strip_local_obs_metadata(obs)
    assert context == {"subgoal": "exit sign"}
    assert "instruction_context" not in obs[0]
    assert serialize_obs(obs)
