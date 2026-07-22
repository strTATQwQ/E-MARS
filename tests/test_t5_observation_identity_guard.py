from __future__ import annotations

import ast
import copy
import sys
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "internvla_ros2"))

from internvla_ros2.observation_guard import (  # noqa: E402
    ObservationStampGate,
    describe_request_identity,
)
from internvla_ros2.protocol import (  # noqa: E402
    ProtocolError,
    RequestIdentity,
    STATUS_OBSERVATION_MISSING,
    STATUS_STALE,
    STATUS_TIMEOUT,
)


MODEL = ROOT / "internvla_ros2/internvla_ros2/model_node.py"
CLIENT = ROOT / "internvla_ros2/internvla_ros2/client_node.py"


@dataclass(frozen=True)
class _Entry:
    revision: int
    message: Any


@dataclass(frozen=True)
class _AcceptedTuple:
    identity: RequestIdentity
    rgb_revision: int
    depth_revision: int
    metadata_revision: int
    epoch: int = 0


class _ObservableCondition:
    def __init__(self) -> None:
        self._condition = threading.Condition()
        self.wait_events = [threading.Event() for _ in range(4)]
        self.wait_count = 0

    def __enter__(self) -> _ObservableCondition:
        self._condition.acquire()
        return self

    def __exit__(self, *_args: object) -> None:
        self._condition.release()

    def wait(self, timeout: float) -> bool:
        index = self.wait_count
        self.wait_count += 1
        if index < len(self.wait_events):
            self.wait_events[index].set()
        return self._condition.wait(timeout)

    def notify_all(self) -> None:
        self._condition.notify_all()


class _ControlledClock:
    def __init__(self, nanoseconds: int) -> None:
        self.nanoseconds = nanoseconds

    def now(self) -> SimpleNamespace:
        return SimpleNamespace(nanoseconds=self.nanoseconds)


class _ModelWaitHarness:
    def __init__(self, clock_ns: int = 100) -> None:
        self._cache_condition = _ObservableCondition()
        self._rgb_cache: dict[int, _Entry] = {}
        self._depth_cache: dict[int, _Entry] = {}
        self._metadata_cache: dict[int, _Entry] = {}
        self._accepted_observation_tuples: OrderedDict[
            int, _AcceptedTuple
        ] = OrderedDict()
        self._observation_cache_size = 32
        self._observation_epoch = 0
        self.clock = _ControlledClock(clock_ns)
        self._t5_sim_time_lock = threading.Lock()
        self._t5_sim_time_high_water_ns = 0

    def get_clock(self) -> _ControlledClock:
        return self.clock

    def _t5_semantic_sim_ns(
        self, observed_ns: int, request_floor_ns: int = 0
    ) -> int:
        observed_ns = int(observed_ns)
        request_floor_ns = int(request_floor_ns)
        with self._t5_sim_time_lock:
            if observed_ns <= 0:
                raise ProtocolError(STATUS_TIMEOUT, "ROS simulation clock is zero")
            if observed_ns < self._t5_sim_time_high_water_ns:
                raise ProtocolError(STATUS_STALE, "ROS simulation clock regressed")
            self._t5_sim_time_high_water_ns = observed_ns
        return max(observed_ns, request_floor_ns)


class _ClientFailure(RuntimeError):
    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = int(status_code)


class _SequenceClock:
    def __init__(self, values: list[int]) -> None:
        self.values = iter(values)
        self.last = values[-1]

    def now(self) -> SimpleNamespace:
        return SimpleNamespace(nanoseconds=next(self.values, self.last))


class _FakeTime:
    def __init__(self, monotonic_values: list[int]) -> None:
        self.values = iter(monotonic_values)
        self.last = monotonic_values[-1]

    def monotonic_ns(self) -> int:
        return next(self.values, self.last)

    def sleep(self, _seconds: float) -> None:
        return None


def _class_method(path: Path, class_name: str, method_name: str) -> ast.FunctionDef:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    return next(
        node
        for node in class_node.body
        if isinstance(node, ast.FunctionDef) and node.name == method_name
    )


def _compiled_method(
    path: Path,
    class_name: str,
    method_name: str,
    namespace: dict[str, Any],
) -> Any:
    method = copy.deepcopy(_class_method(path, class_name, method_name))
    method.decorator_list = []
    module = ast.Module(body=[method], type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[method_name]


def _metadata(episode: str, generation: int, sequence: int, request: str) -> Any:
    return SimpleNamespace(
        episode_id=episode,
        reset_generation=generation,
        sequence_id=sequence,
        request_id=request,
    )


def _metadata_identity(metadata: Any) -> RequestIdentity:
    return RequestIdentity(
        str(metadata.episode_id),
        int(metadata.reset_generation),
        int(metadata.sequence_id),
        str(metadata.request_id),
    )


def _mismatch_message(
    expected: RequestIdentity, observed: RequestIdentity
) -> str:
    return (
        "observation metadata identity mismatch: "
        f"expected={describe_request_identity(expected)} "
        f"observed={describe_request_identity(observed)}"
    )


def _model_wait_method() -> Any:
    return _compiled_method(
        MODEL,
        "InternVLAModelNode",
        "_wait_for_observation",
        {
            "Any": Any,
            "ObservationMetadata": object,
            "RequestIdentity": RequestIdentity,
            "_AcceptedObservationTuple": _AcceptedTuple,
            "_metadata_identity": _metadata_identity,
            "_identity_mismatch_message": _mismatch_message,
            "describe_request_identity": describe_request_identity,
            "ProtocolError": ProtocolError,
            "STATUS_OBSERVATION_MISSING": STATUS_OBSERVATION_MISSING,
            "STATUS_STALE": STATUS_STALE,
            "STATUS_TIMEOUT": STATUS_TIMEOUT,
            "time": __import__("time"),
        },
    )


def _model_mark_method() -> Any:
    return _compiled_method(
        MODEL,
        "InternVLAModelNode",
        "_mark_observation_tuple_accepted",
        {
            "_AcceptedObservationTuple": _AcceptedTuple,
            "ProtocolError": ProtocolError,
            "STATUS_STALE": STATUS_STALE,
            "describe_request_identity": describe_request_identity,
        },
    )


def _sim_time_method(path: Path, class_name: str, namespace: dict[str, Any]) -> Any:
    return _compiled_method(
        path,
        class_name,
        "_t5_semantic_sim_ns",
        namespace,
    )


def _run_in_thread(function: Any, *args: Any) -> tuple[threading.Thread, dict[str, Any]]:
    outcome: dict[str, Any] = {}

    def target() -> None:
        try:
            outcome["value"] = function(*args)
        except BaseException as exc:  # assertion inspects the exact typed failure
            outcome["error"] = exc

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    return thread, outcome


def test_model_wait_binds_identity_and_waits_past_stale_same_stamp() -> None:
    method = _class_method(MODEL, "InternVLAModelNode", "_wait_for_observation")
    source = ast.unparse(method)

    assert "identity: RequestIdentity" in source
    assert "observed_identity = _metadata_identity(metadata)" in source
    assert "if observed_identity != identity" in source
    assert "stale_identity = observed_identity" in source
    assert "expected_epoch: int" in source
    assert "if self._observation_epoch != expected_epoch" in source
    assert "stale_tuple_baseline = _AcceptedObservationTuple" in source
    assert "baseline = self._accepted_observation_tuples.get(stamp_ns)" in source
    assert "accepted = stale_tuple_baseline" in source
    assert "tuple_revisions.rgb_revision > accepted.rgb_revision" in source
    assert "tuple_revisions.depth_revision > accepted.depth_revision" in source
    assert "tuple_revisions.metadata_revision > accepted.metadata_revision" in source
    assert "self._cache_condition.wait" in source
    assert "STATUS_STALE" in source
    assert "_identity_mismatch_message(identity, stale_identity)" in source


def test_execute_captures_request_epoch_before_initial_barrier_validation() -> None:
    method = _class_method(MODEL, "InternVLAModelNode", "_execute_step")
    source = ast.unparse(method)

    epoch_capture = source.index("request_epoch = self._observation_epoch")
    first_barrier = source.index("self._barrier.validate_current(identity)")
    wait_binding = source.index("expected_epoch=request_epoch")
    assert "with self._inference_lock" in source[:first_barrier]
    assert epoch_capture < first_barrier < wait_binding


def test_model_waits_for_matching_identity_and_all_new_component_revisions() -> None:
    stamp = 100
    old = RequestIdentity("b::121", 4, 0, "b::121:4:0")
    expected = RequestIdentity("b::121", 4, 1, "b::121:4:1")
    harness = _ModelWaitHarness(stamp)
    harness._rgb_cache[stamp] = _Entry(1, "old-rgb")
    harness._depth_cache[stamp] = _Entry(2, "old-depth")
    harness._metadata_cache[stamp] = _Entry(
        3, _metadata("b::121", 4, 0, "b::121:4:0")
    )
    harness._accepted_observation_tuples[stamp] = _AcceptedTuple(old, 1, 2, 3)

    thread, outcome = _run_in_thread(
        _model_wait_method(),
        harness,
        stamp,
        expected,
        harness._observation_epoch,
        stamp + 10_000_000_000,
        __import__("time").monotonic_ns() + 10_000_000_000,
    )
    assert harness._cache_condition.wait_events[0].wait(1.0)
    assert thread.is_alive()

    # Matching metadata arriving first must not pair with the old images.
    with harness._cache_condition:
        harness._metadata_cache[stamp] = _Entry(
            6, _metadata("b::121", 4, 1, "b::121:4:1")
        )
        harness._cache_condition.notify_all()
    assert harness._cache_condition.wait_events[1].wait(1.0)
    assert thread.is_alive()

    with harness._cache_condition:
        harness._rgb_cache[stamp] = _Entry(7, "new-rgb")
        harness._depth_cache[stamp] = _Entry(8, "new-depth")
        harness._cache_condition.notify_all()
    thread.join(1.0)

    assert not thread.is_alive()
    assert "error" not in outcome
    rgb, depth, metadata, revisions, _last_sim_ns = outcome["value"]
    assert (rgb, depth) == ("new-rgb", "new-depth")
    assert _metadata_identity(metadata) == expected
    assert revisions == _AcceptedTuple(expected, 7, 8, 6)


def test_model_wait_uses_observed_stale_tuple_as_baseline_before_acceptance() -> None:
    stamp = 100
    expected = RequestIdentity("b::121", 4, 1, "b::121:4:1")
    harness = _ModelWaitHarness(stamp)
    harness._rgb_cache[stamp] = _Entry(1, "old-rgb")
    harness._depth_cache[stamp] = _Entry(2, "old-depth")
    harness._metadata_cache[stamp] = _Entry(
        3, _metadata("b::121", 4, 0, "b::121:4:0")
    )

    thread, outcome = _run_in_thread(
        _model_wait_method(),
        harness,
        stamp,
        expected,
        harness._observation_epoch,
        stamp + 10_000_000_000,
        __import__("time").monotonic_ns() + 10_000_000_000,
    )
    assert harness._cache_condition.wait_events[0].wait(1.0)
    with harness._cache_condition:
        harness._metadata_cache[stamp] = _Entry(
            4, _metadata("b::121", 4, 1, "b::121:4:1")
        )
        harness._cache_condition.notify_all()
    assert harness._cache_condition.wait_events[1].wait(1.0)
    assert thread.is_alive()

    with harness._cache_condition:
        harness._rgb_cache[stamp] = _Entry(5, "new-rgb")
        harness._depth_cache[stamp] = _Entry(6, "new-depth")
        harness._cache_condition.notify_all()
    thread.join(1.0)

    assert not thread.is_alive()
    assert "error" not in outcome
    assert outcome["value"][0:2] == ("new-rgb", "new-depth")


def test_delayed_old_components_raise_stale_baseline_before_new_metadata() -> None:
    stamp = 100
    expected = RequestIdentity("b::121", 4, 1, "b::121:4:1")
    harness = _ModelWaitHarness(stamp)
    harness._metadata_cache[stamp] = _Entry(
        1, _metadata("b::121", 4, 0, "b::121:4:0")
    )

    thread, outcome = _run_in_thread(
        _model_wait_method(),
        harness,
        stamp,
        expected,
        harness._observation_epoch,
        stamp + 10_000_000_000,
        __import__("time").monotonic_ns() + 10_000_000_000,
    )
    assert harness._cache_condition.wait_events[0].wait(1.0)

    # These are delayed components belonging to the still-current stale
    # metadata. They must become part of the barrier, not satisfy it.
    with harness._cache_condition:
        harness._rgb_cache[stamp] = _Entry(2, "delayed-old-rgb")
        harness._depth_cache[stamp] = _Entry(3, "delayed-old-depth")
        harness._cache_condition.notify_all()
    assert harness._cache_condition.wait_events[1].wait(1.0)

    with harness._cache_condition:
        harness._metadata_cache[stamp] = _Entry(
            4, _metadata("b::121", 4, 1, "b::121:4:1")
        )
        harness._cache_condition.notify_all()
    assert harness._cache_condition.wait_events[2].wait(1.0)
    assert thread.is_alive()

    with harness._cache_condition:
        harness._rgb_cache[stamp] = _Entry(5, "new-rgb")
        harness._depth_cache[stamp] = _Entry(6, "new-depth")
        harness._cache_condition.notify_all()
    thread.join(1.0)

    assert not thread.is_alive()
    assert "error" not in outcome
    assert outcome["value"][0:2] == ("new-rgb", "new-depth")


def test_acceptance_mark_monotonically_merges_same_identity_revisions() -> None:
    stamp = 100
    identity = RequestIdentity("b::121", 4, 1, "b::121:4:1")
    harness = _ModelWaitHarness(stamp)
    harness._accepted_observation_tuples[stamp] = _AcceptedTuple(
        identity, 10, 20, 30
    )

    _model_mark_method()(harness, stamp, _AcceptedTuple(identity, 12, 18, 35))

    assert harness._accepted_observation_tuples[stamp] == _AcceptedTuple(
        identity, 12, 20, 35
    )


def test_late_regressing_mark_fails_closed_and_preserves_newer_baseline() -> None:
    stamp = 100
    current_identity = RequestIdentity("b::121", 4, 1, "b::121:4:1")
    late_identity = RequestIdentity("b::121", 4, 0, "b::121:4:0")
    current = _AcceptedTuple(current_identity, 10, 20, 30)
    harness = _ModelWaitHarness(stamp)
    harness._accepted_observation_tuples[stamp] = current

    try:
        _model_mark_method()(
            harness, stamp, _AcceptedTuple(late_identity, 11, 21, 29)
        )
    except ProtocolError as exc:
        assert exc.status_code == STATUS_STALE
        assert "revision regression" in str(exc)
        assert "current_identity={episode_id='b::121'" in str(exc)
        assert "candidate_identity={episode_id='b::121'" in str(exc)
    else:
        raise AssertionError("regressing acceptance mark was not rejected")

    assert harness._accepted_observation_tuples[stamp] == current


def test_pre_reset_late_mark_fails_closed_and_cannot_repopulate_cleared_ledger() -> None:
    stamp = 100
    identity = RequestIdentity("b::121", 4, 1, "b::121:4:1")
    harness = _ModelWaitHarness(stamp)
    candidate = _AcceptedTuple(identity, 10, 20, 30, epoch=0)

    # Simulate reset clearing every observation ledger after wait returned but
    # before the old request reached its acceptance mark.
    harness._accepted_observation_tuples.clear()
    harness._observation_epoch = 1
    try:
        _model_mark_method()(harness, stamp, candidate)
    except ProtocolError as exc:
        assert exc.status_code == STATUS_STALE
        assert "cache epoch changed" in str(exc)
        assert "candidate_epoch=0" in str(exc)
        assert "current_epoch=1" in str(exc)
    else:
        raise AssertionError("pre-reset tuple repopulated the cleared ledger")

    assert stamp not in harness._accepted_observation_tuples


def test_new_identity_mark_requires_every_component_revision_to_advance() -> None:
    stamp = 100
    first = RequestIdentity("b::121", 4, 0, "b::121:4:0")
    second = RequestIdentity("b::121", 4, 1, "b::121:4:1")
    harness = _ModelWaitHarness(stamp)
    harness._accepted_observation_tuples[stamp] = _AcceptedTuple(first, 1, 2, 3)

    _model_mark_method()(harness, stamp, _AcceptedTuple(second, 4, 5, 6))

    assert harness._accepted_observation_tuples[stamp] == _AcceptedTuple(
        second, 4, 5, 6
    )


def test_model_stale_timeout_reports_expected_and_observed_identity() -> None:
    stamp = 100
    expected = RequestIdentity("b::121", 4, 1, "b::121:4:1")
    harness = _ModelWaitHarness(stamp)
    harness._rgb_cache[stamp] = _Entry(1, "old-rgb")
    harness._depth_cache[stamp] = _Entry(2, "old-depth")
    harness._metadata_cache[stamp] = _Entry(
        3, _metadata("b::121", 4, 0, "b::121:4:0")
    )

    try:
        _model_wait_method()(
            harness, stamp, expected, harness._observation_epoch, stamp, 10**30
        )
    except ProtocolError as exc:
        assert exc.status_code == STATUS_STALE
        assert "expected={episode_id='b::121', reset_generation=4" in str(exc)
        assert "sequence_id=1, request_id='b::121:4:1'" in str(exc)
        assert "observed={episode_id='b::121', reset_generation=4" in str(exc)
        assert "sequence_id=0, request_id='b::121:4:0'" in str(exc)
    else:
        raise AssertionError("stale same-stamp metadata was not rejected")


def test_recovery_preserved_cache_rejects_old_tuple_then_accepts_exact_retry() -> None:
    stamp = 100
    prior = RequestIdentity("a::259", 1, 12, "a::259:1:12")
    expected = RequestIdentity("a::259", 1, 13, "a::259:1:13")
    prior_digest = "11" * 32
    expected_digest = "22" * 32
    harness = _ModelWaitHarness(stamp)
    prior_metadata = _metadata(
        prior.episode_id,
        prior.reset_generation,
        prior.sequence_id,
        prior.request_id,
    )
    prior_metadata.observation_digest = prior_digest
    harness._rgb_cache[stamp] = _Entry(1, "prior-rgb")
    harness._depth_cache[stamp] = _Entry(2, "prior-depth")
    harness._metadata_cache[stamp] = _Entry(3, prior_metadata)

    # A typed completion_sim history clear deliberately leaves this cache and
    # epoch intact. The next identity must not consume the retained old tuple,
    # even when DDS reuses the same simulator stamp.
    try:
        _model_wait_method()(
            harness,
            stamp,
            expected,
            harness._observation_epoch,
            stamp,
            10**30,
        )
    except ProtocolError as exc:
        assert exc.status_code == STATUS_STALE
        assert prior.request_id in str(exc)
        assert expected.request_id in str(exc)
    else:
        raise AssertionError("retained pre-clear observation crossed identity")

    retry_metadata = _metadata(
        expected.episode_id,
        expected.reset_generation,
        expected.sequence_id,
        expected.request_id,
    )
    retry_metadata.observation_digest = expected_digest
    with harness._cache_condition:
        harness._rgb_cache[stamp] = _Entry(4, "retry-rgb")
        harness._depth_cache[stamp] = _Entry(5, "retry-depth")
        harness._metadata_cache[stamp] = _Entry(6, retry_metadata)

    rgb, depth, metadata, revisions, _ = _model_wait_method()(
        harness,
        stamp,
        expected,
        harness._observation_epoch,
        stamp + 1,
        10**30,
    )

    assert (rgb, depth) == ("retry-rgb", "retry-depth")
    assert _metadata_identity(metadata) == expected
    assert metadata.observation_digest == expected_digest
    assert revisions == _AcceptedTuple(expected, 4, 5, 6, epoch=0)


def test_matching_metadata_with_missing_component_forgets_prior_stale_identity() -> None:
    stamp = 100
    expected = RequestIdentity("b::121", 4, 1, "b::121:4:1")
    harness = _ModelWaitHarness(stamp)
    harness._rgb_cache[stamp] = _Entry(1, "old-rgb")
    harness._depth_cache[stamp] = _Entry(2, "old-depth")
    harness._metadata_cache[stamp] = _Entry(
        3, _metadata("b::121", 4, 0, "b::121:4:0")
    )

    thread, outcome = _run_in_thread(
        _model_wait_method(),
        harness,
        stamp,
        expected,
        harness._observation_epoch,
        stamp + 10,
        10**30,
    )
    assert harness._cache_condition.wait_events[0].wait(1.0)
    with harness._cache_condition:
        harness._metadata_cache[stamp] = _Entry(
            4, _metadata("b::121", 4, 1, "b::121:4:1")
        )
        harness._depth_cache.pop(stamp)
        harness.clock.nanoseconds = stamp + 10
        harness._cache_condition.notify_all()
    thread.join(1.0)

    assert not thread.is_alive()
    error = outcome.get("error")
    assert isinstance(error, ProtocolError)
    assert error.status_code == STATUS_OBSERVATION_MISSING
    assert "b::121:4:0" not in str(error)


def test_model_wait_fails_closed_on_zero_or_regressed_sim_clock() -> None:
    identity = RequestIdentity("b::121", 4, 1, "b::121:4:1")
    method = _model_wait_method()

    for observed_ns, expected_status in ((0, STATUS_TIMEOUT), (99, STATUS_STALE)):
        harness = _ModelWaitHarness(observed_ns)
        if expected_status == STATUS_STALE:
            harness._t5_sim_time_high_water_ns = 100
        try:
            method(
                harness,
                100,
                identity,
                harness._observation_epoch,
                1_000,
                10**30,
                100,
            )
        except ProtocolError as exc:
            assert exc.status_code == expected_status
            assert "simulation clock" in str(exc)
        else:
            raise AssertionError("invalid simulator clock did not fail closed")


def test_model_wait_uses_sim_stamp_for_observation_semantic_deadline() -> None:
    identity = RequestIdentity("b::121", 4, 1, "b::121:4:1")
    harness = _ModelWaitHarness(101)

    try:
        _model_wait_method()(
            harness,
            100,
            identity,
            harness._observation_epoch,
            100,
            10**30,
            100,
        )
    except ProtocolError as exc:
        assert exc.status_code == STATUS_OBSERVATION_MISSING
        assert "simulation-time observation deadline" in str(exc)
    else:
        raise AssertionError("expired simulator observation deadline was ignored")


def test_wait_fails_stale_when_reset_changes_epoch_before_old_tuple_arrives() -> None:
    stamp = 100
    expected = RequestIdentity("b::121", 4, 1, "b::121:4:1")
    harness = _ModelWaitHarness(stamp)

    thread, outcome = _run_in_thread(
        _model_wait_method(),
        harness,
        stamp,
        expected,
        harness._observation_epoch,
        stamp + 10_000_000_000,
        __import__("time").monotonic_ns() + 10_000_000_000,
    )
    assert harness._cache_condition.wait_events[0].wait(1.0)

    # Reset clears every cache and advances the epoch while the old request is
    # blocked. Even if its matching tuple arrives before it reacquires the
    # condition, that pre-reset request must not consume or ledger it.
    with harness._cache_condition:
        harness._rgb_cache.clear()
        harness._depth_cache.clear()
        harness._metadata_cache.clear()
        harness._accepted_observation_tuples.clear()
        harness._observation_epoch += 1
        harness._rgb_cache[stamp] = _Entry(10, "post-reset-rgb")
        harness._depth_cache[stamp] = _Entry(11, "post-reset-depth")
        harness._metadata_cache[stamp] = _Entry(
            12, _metadata("b::121", 4, 1, "b::121:4:1")
        )
        harness._cache_condition.notify_all()
    thread.join(1.0)

    assert not thread.is_alive()
    error = outcome.get("error")
    assert isinstance(error, ProtocolError)
    assert error.status_code == STATUS_STALE
    assert "cache epoch changed while waiting" in str(error)
    assert "expected_epoch=0" in str(error)
    assert "current_epoch=1" in str(error)
    assert not harness._accepted_observation_tuples


def test_wait_rejects_reset_between_barrier_validation_and_wait_entry() -> None:
    stamp = 100
    expected = RequestIdentity("b::121", 4, 1, "b::121:4:1")
    harness = _ModelWaitHarness(stamp)
    request_epoch = harness._observation_epoch

    # The request already captured epoch 0 and passed its first barrier check.
    # Reset then completes before `_wait_for_observation` acquires the cache
    # condition, and an old matching tuple arrives in the new cache epoch.
    with harness._cache_condition:
        harness._rgb_cache.clear()
        harness._depth_cache.clear()
        harness._metadata_cache.clear()
        harness._accepted_observation_tuples.clear()
        harness._observation_epoch += 1
        harness._rgb_cache[stamp] = _Entry(10, "old-request-rgb")
        harness._depth_cache[stamp] = _Entry(11, "old-request-depth")
        harness._metadata_cache[stamp] = _Entry(
            12, _metadata("b::121", 4, 1, "b::121:4:1")
        )

    try:
        _model_wait_method()(
            harness,
            stamp,
            expected,
            request_epoch,
            stamp + 10_000_000_000,
            10**30,
        )
    except ProtocolError as exc:
        assert exc.status_code == STATUS_STALE
        assert "cache epoch changed before wait" in str(exc)
        assert "expected_epoch=0" in str(exc)
        assert "current_epoch=1" in str(exc)
    else:
        raise AssertionError("old request inherited the post-reset cache epoch")

    assert not harness._accepted_observation_tuples


def test_model_mismatch_diagnostic_keeps_expected_and_observed_identity() -> None:
    source = MODEL.read_text(encoding="utf-8")

    assert "expected={describe_request_identity(expected)}" in source
    assert "observed={describe_request_identity(observed)}" in source
    assert "raise ProtocolError(\n                STATUS_STALE," in source
    assert "warning" not in ast.unparse(
        _class_method(MODEL, "InternVLAModelNode", "_validate_metadata")
    ).lower()


def test_client_stamp_wait_is_wall_bounded_and_safe_stops_on_timeout() -> None:
    method = _class_method(
        CLIENT, "InternVLAClientNode", "_wait_for_observation_stamp"
    )
    source = ast.unparse(method)

    assert "time.monotonic_ns()" in source
    assert "self._observation_stamp_gate.accepts(identity, observed_stamp_ns)" in source
    assert "self._safe_stop(STATUS_TIMEOUT, message)" in source
    assert "raise ClientFailure(STATUS_TIMEOUT, message)" in source
    assert "observed_stamp_ns + 1" not in source


def test_t5_sim_time_only_guard_is_exact_on_both_step_nodes() -> None:
    for path, class_name in (
        (CLIENT, "InternVLAClientNode"),
        (MODEL, "InternVLAModelNode"),
    ):
        source = ast.unparse(_class_method(path, class_name, "__init__"))
        assert "INTERNNAV_RUNTIME_POLICY" in source
        assert "completion_sim" in source
        assert "INTERNNAV_SIMULATION_TARGET" in source
        assert "isaac" in source
        assert "INTERNNAV_T5_LANE" in source
        assert "{'a', 'b'}" in source
        assert "self.get_parameter('use_sim_time').value" in source
        assert (
            "if self.t5_sim_time_only and "
            "(not bool(self.get_parameter('use_sim_time').value))" in source
        )
        assert "raise RuntimeError('T5 completion_sim requires use_sim_time=true')" in source


def test_t5_step_semantics_exclude_wall_deadline_but_keep_liveness_watchdogs() -> None:
    client_source = ast.unparse(
        _class_method(CLIENT, "InternVLAClientNode", "step_arrays")
    )
    model_source = ast.unparse(
        _class_method(MODEL, "InternVLAModelNode", "_execute_step")
    )
    wait_source = ast.unparse(
        _class_method(MODEL, "InternVLAModelNode", "_wait_for_observation")
    )

    assert "self.service_timeout_sec if self.t5_sim_time_only" in client_source
    assert "if self.t5_sim_time_only and response_sim_ns" not in client_source
    assert "if self.t5_sim_time_only:" in client_source
    assert "response exceeded ROS simulation-time deadline" in client_source
    assert "if not self.t5_sim_time_only" in client_source
    assert "sim_time_only=True" in model_source
    assert "time.monotonic_ns() + observation_wait_ns" in model_source
    assert "wait_monotonic_deadline_ns - time.monotonic_ns()" in wait_source
    assert "self._t5_semantic_sim_ns" in client_source
    assert "self._t5_semantic_sim_ns" in model_source
    assert "self._t5_semantic_sim_ns" in wait_source


def test_t5_sim_time_high_water_persists_across_requests() -> None:
    safe_stops: list[tuple[int, str]] = []
    client = SimpleNamespace(
        _t5_sim_time_lock=threading.Lock(),
        _t5_sim_time_high_water_ns=0,
        _safe_stop=lambda code, message: safe_stops.append((code, message)),
    )
    client_method = _sim_time_method(
        CLIENT,
        "InternVLAClientNode",
        {
            "ClientFailure": _ClientFailure,
            "STATUS_STALE": STATUS_STALE,
            "STATUS_TIMEOUT": STATUS_TIMEOUT,
        },
    )
    assert client_method(client, 10_000) == 10_000
    try:
        client_method(client, 9_700)
    except _ClientFailure as exc:
        assert exc.status_code == STATUS_STALE
    else:
        raise AssertionError("client accepted a cross-request sim-time regression")
    assert client._t5_sim_time_high_water_ns == 10_000
    assert client_method(client, 10_000) == 10_000
    assert safe_stops[-1][0] == STATUS_STALE

    model = SimpleNamespace(
        _t5_sim_time_lock=threading.Lock(),
        _t5_sim_time_high_water_ns=0,
    )
    model_method = _sim_time_method(
        MODEL,
        "InternVLAModelNode",
        {
            "ProtocolError": ProtocolError,
            "STATUS_STALE": STATUS_STALE,
            "STATUS_TIMEOUT": STATUS_TIMEOUT,
        },
    )
    assert model_method(model, 10_000) == 10_000
    try:
        model_method(model, 9_700)
    except ProtocolError as exc:
        assert exc.status_code == STATUS_STALE
    else:
        raise AssertionError("model accepted a cross-request sim-time regression")
    assert model._t5_sim_time_high_water_ns == 10_000
    assert model_method(model, 10_000) == 10_000


def test_model_receiver_lag_one_tick_uses_request_stamp_as_semantic_floor() -> None:
    model = SimpleNamespace(
        _t5_sim_time_lock=threading.Lock(),
        _t5_sim_time_high_water_ns=9_998,
    )
    model_method = _sim_time_method(
        MODEL,
        "InternVLAModelNode",
        {
            "ProtocolError": ProtocolError,
            "STATUS_STALE": STATUS_STALE,
            "STATUS_TIMEOUT": STATUS_TIMEOUT,
        },
    )

    # The request was stamped on the client's next /clock tick, but the model
    # receiver has only observed the preceding tick locally.  This is not a
    # rollback: semantic validation starts at the request stamp while the
    # persistent high-water remains the actual local observation.
    assert model_method(model, 9_999, 10_000) == 10_000
    assert model._t5_sim_time_high_water_ns == 9_999
    assert model_method(model, 10_000, 10_000) == 10_000
    assert model._t5_sim_time_high_water_ns == 10_000

    source = ast.unparse(_class_method(MODEL, "InternVLAModelNode", "_execute_step"))
    assert "self._t5_semantic_sim_ns(receive_sim_ns, sim_stamp_ns)" in source
    assert "self._t5_semantic_sim_ns(wait_sim_ns, sim_stamp_ns)" in source
    assert "self._t5_semantic_sim_ns(feedback_now.nanoseconds, sim_stamp_ns)" in source
    assert "self._t5_semantic_sim_ns(now_sim_ns, sim_stamp_ns)" in source
    assert "self._t5_semantic_sim_ns(now_ns, sim_stamp_ns)" in source


def test_client_stamp_wait_rejects_equal_and_regressed_until_clock_advances() -> None:
    identity = RequestIdentity("b::121", 4, 1, "b::121:4:1")
    safe_stops: list[tuple[int, str]] = []
    clock = _SequenceClock([100, 99, 101])
    harness = SimpleNamespace(
        observation_stamp_wait_sec=1.0,
        _observation_stamp_gate=ObservationStampGate(
            RequestIdentity("b::121", 4, 0, "b::121:4:0"), 100
        ),
        get_clock=lambda: clock,
        _safe_stop=lambda code, message: safe_stops.append((code, message)),
    )
    fake_time = _FakeTime([0, 1, 2])
    method = _compiled_method(
        CLIENT,
        "InternVLAClientNode",
        "_wait_for_observation_stamp",
        {
            "Any": Any,
            "RequestIdentity": RequestIdentity,
            "STATUS_TIMEOUT": STATUS_TIMEOUT,
            "ClientFailure": _ClientFailure,
            "describe_request_identity": describe_request_identity,
            "time": fake_time,
        },
    )

    selected = method(harness, identity, 1.0)

    assert selected.nanoseconds == 101
    assert harness._observation_stamp_gate.last_stamp_ns == 101
    assert harness._observation_stamp_gate.last_identity == identity
    assert safe_stops == []


def test_client_frozen_stamp_timeout_safe_stops_without_advancing_gate() -> None:
    previous = RequestIdentity("b::121", 4, 0, "b::121:4:0")
    requested = RequestIdentity("b::121", 4, 1, "b::121:4:1")
    gate = ObservationStampGate(previous, 100)
    safe_stops: list[tuple[int, str]] = []
    clock = _SequenceClock([100])
    harness = SimpleNamespace(
        observation_stamp_wait_sec=1.0,
        _observation_stamp_gate=gate,
        get_clock=lambda: clock,
        _safe_stop=lambda code, message: safe_stops.append((code, message)),
    )
    method = _compiled_method(
        CLIENT,
        "InternVLAClientNode",
        "_wait_for_observation_stamp",
        {
            "Any": Any,
            "RequestIdentity": RequestIdentity,
            "STATUS_TIMEOUT": STATUS_TIMEOUT,
            "ClientFailure": _ClientFailure,
            "describe_request_identity": describe_request_identity,
            "time": _FakeTime([0, 2_000_000_000]),
        },
    )

    try:
        method(harness, requested, 1.0)
    except _ClientFailure as exc:
        assert exc.status_code == STATUS_TIMEOUT
        assert "previous_stamp_ns=100" in str(exc)
        assert "observed_stamp_ns=100" in str(exc)
        assert "requested_identity={episode_id='b::121'" in str(exc)
    else:
        raise AssertionError("frozen simulator stamp did not time out")

    assert safe_stops and safe_stops[0][0] == STATUS_TIMEOUT
    assert gate.last_identity == previous
    assert gate.last_stamp_ns == 100


def test_client_uses_guard_before_publishing_observation_tuple() -> None:
    method = _class_method(CLIENT, "InternVLAClientNode", "step_arrays")
    calls = {
        node.func.attr: node.lineno
        for node in ast.walk(method)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        and node.func.attr
        in {"_wait_for_observation_stamp", "publish"}
    }

    assert calls["_wait_for_observation_stamp"] < calls["publish"]
