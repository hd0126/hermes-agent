"""Freshness-safe repeated read guardrail tests."""

from agent.tool_guardrails import ToolCallGuardrailConfig, ToolCallGuardrailController


def _fresh_controller(**overrides):
    return ToolCallGuardrailController(
        ToolCallGuardrailConfig(
            warnings_enabled=False,
            hard_stop_enabled=True,
            freshness_safe_reads=True,
            **overrides,
        )
    )


def test_config_unchanged_keeps_warning_only_static_reads_unbounded_by_hard_stop_default():
    controller = ToolCallGuardrailController(
        ToolCallGuardrailConfig(no_progress_warn_after=2, no_progress_block_after=2)
    )
    args = {"path": "same.txt"}

    decisions = []
    for _ in range(6):
        assert controller.before_call("read_file", args).action == "allow"
        decisions.append(controller.after_call("read_file", args, "same", failed=False))

    assert [d.action for d in decisions] == ["allow", "warn", "warn", "warn", "warn", "warn"]
    assert controller.before_call("read_file", args).action == "allow"
    assert controller.halt_decision is None


def test_freshness_safe_reads_reuses_legacy_idempotent_hard_stop_threshold():
    cfg = ToolCallGuardrailConfig.from_mapping(
        {
            "freshness_safe_reads": True,
            "hard_stop_after": {"idempotent_no_progress": 3, "poll_no_progress": 14},
        }
    )

    assert cfg.freshness_safe_reads is True
    assert cfg.no_progress_block_after == 3
    assert cfg.poll_no_progress_block_after == 14


def test_freshness_safe_reads_without_hard_stop_keeps_warning_only_contract():
    controller = ToolCallGuardrailController(
        ToolCallGuardrailConfig(
            warnings_enabled=True,
            hard_stop_enabled=False,
            freshness_safe_reads=True,
            no_progress_warn_after=2,
            no_progress_block_after=2,
        )
    )
    args = {"path": "same.txt"}

    decisions = []
    for _ in range(4):
        assert controller.before_call("read_file", args).action == "allow"
        decisions.append(controller.after_call("read_file", args, "same", failed=False))

    assert [d.action for d in decisions] == ["allow", "warn", "warn", "warn"]
    assert controller.halt_decision is None


def test_freshness_safe_disabled_preserves_skill_view_as_legacy_untracked():
    controller = ToolCallGuardrailController(
        ToolCallGuardrailConfig(
            warnings_enabled=True,
            hard_stop_enabled=True,
            freshness_safe_reads=False,
            no_progress_warn_after=2,
            no_progress_block_after=2,
        )
    )
    args = {"name": "example"}

    for _ in range(4):
        assert controller.before_call("skill_view", args).action == "allow"
        decision = controller.after_call("skill_view", args, "same", failed=False)
        assert decision.action == "allow"
        assert decision.code == "allow"
    assert controller.halt_decision is None


def test_freshness_safe_tracks_skill_view_repeated_success_when_enabled():
    controller = _fresh_controller(no_progress_block_after=2)
    args = {"name": "example"}

    assert controller.before_call("skill_view", args).action == "allow"
    assert controller.after_call("skill_view", args, "same", failed=False).action == "allow"
    assert controller.before_call("skill_view", args).action == "allow"
    halted = controller.after_call("skill_view", args, "same", failed=False)
    assert halted.action == "halt"
    assert halted.code == "idempotent_repeated_success_halt"


def test_freshness_safe_reads_disabled_preserves_legacy_pre_dispatch_static_read_block():
    controller = ToolCallGuardrailController(
        ToolCallGuardrailConfig(
            warnings_enabled=False,
            hard_stop_enabled=True,
            freshness_safe_reads=False,
            no_progress_block_after=2,
        )
    )
    args = {"path": "same.txt"}

    for _ in range(2):
        assert controller.before_call("read_file", args).action == "allow"
        assert controller.after_call("read_file", args, "same", failed=False).action == "allow"

    blocked = controller.before_call("read_file", args)
    assert blocked.action == "block"
    assert blocked.code == "idempotent_no_progress_block"


def test_freshness_safe_changed_fifth_static_read_is_allowed_and_resets_streak():
    controller = _fresh_controller(no_progress_block_after=5)
    args = {"path": "changing.txt"}

    for _ in range(4):
        assert controller.before_call("read_file", args).action == "allow"
        assert controller.after_call("read_file", args, "old", failed=False).action == "allow"

    assert controller.before_call("read_file", args).action == "allow"
    changed = controller.after_call("read_file", args, "new", failed=False)
    assert changed.action == "allow"
    assert changed.count == 1
    assert controller.halt_decision is None


def test_successful_edit_resets_exact_and_aggregate_failure_streaks():
    controller = _fresh_controller(exact_failure_block_after=2, same_tool_failure_halt_after=3)
    args = {"command": "run-checks"}
    for i in range(12):
        assert controller.before_call("terminal", args).allows_execution
        result = controller.after_call("terminal", args, '{"exit_code":1}', failed=True)
        assert not result.should_halt
        controller.after_call("patch", {"path": "example.py", "revision": i}, '{"success":true}', failed=False)
    assert controller.before_call("terminal", args).allows_execution
    assert controller.halt_decision is None


def test_different_failed_diagnostic_commands_do_not_halt_in_safe_mode():
    controller = _fresh_controller(same_tool_failure_halt_after=3)
    for i in range(12):
        result = controller.after_call("terminal", {"command": f"diagnostic-{i}"}, '{"exit_code":1}', failed=True)
        assert not result.should_halt
    assert controller.halt_decision is None


def test_unchanged_failed_retry_is_still_blocked_without_progress():
    controller = _fresh_controller(exact_failure_block_after=2)
    args = {"command": "same-failing-check"}
    for _ in range(2):
        assert controller.before_call("terminal", args).allows_execution
        controller.after_call("terminal", args, '{"exit_code":1}', failed=True)
    assert controller.before_call("terminal", args).action == "block"


def test_failed_edit_and_successful_poll_do_not_erase_failures():
    controller = _fresh_controller(exact_failure_block_after=2)
    args = {"command": "same-failing-check"}
    for _ in range(2):
        controller.after_call("terminal", args, '{"exit_code":1}', failed=True)
    controller.after_call("patch", {"path": "example.py"}, '{"error":"not applied"}', failed=True)
    controller.after_call("process", {"action": "poll", "session_id": "example"}, '{"status":"running"}', failed=False)
    controller.after_call("read_file", {"path": "example.py"}, '{"content":"same"}', failed=False)
    assert controller.before_call("terminal", args).action == "block"


def test_legacy_distinct_failure_limit_is_unchanged():
    controller = ToolCallGuardrailController(ToolCallGuardrailConfig(
        hard_stop_enabled=True, freshness_safe_reads=False, same_tool_failure_halt_after=3,
    ))
    for i in range(3):
        decision = controller.after_call("terminal", {"command": f"check-{i}"}, '{"exit_code":1}', failed=True)
    assert decision.code == "same_tool_failure_halt"


def test_freshness_safe_same_fifth_static_read_halts_after_success_and_blocks_next_call():
    controller = _fresh_controller(no_progress_block_after=5)
    args = {"path": "same.txt"}

    decisions = []
    for _ in range(5):
        assert controller.before_call("read_file", args).action == "allow"
        decisions.append(controller.after_call("read_file", args, "same", failed=False))

    assert [d.action for d in decisions[:4]] == ["allow", "allow", "allow", "allow"]
    assert decisions[4].action == "halt"
    assert decisions[4].code == "idempotent_repeated_success_halt"
    blocked = controller.before_call("read_file", args)
    assert blocked.action == "halt"
    assert blocked.code == "idempotent_repeated_success_halt"


def test_freshness_safe_successful_write_invalidates_read_streaks():
    controller = _fresh_controller(no_progress_block_after=5)
    read_args = {"path": "same.txt"}

    for _ in range(4):
        assert controller.before_call("read_file", read_args).action == "allow"
        assert controller.after_call("read_file", read_args, "same", failed=False).action == "allow"

    assert controller.before_call("write_file", {"path": "same.txt", "content": "x"}).action == "allow"
    assert controller.after_call(
        "write_file", {"path": "same.txt", "content": "x"}, "ok", failed=False
    ).action == "allow"

    assert controller.before_call("read_file", read_args).action == "allow"
    decision = controller.after_call("read_file", read_args, "same", failed=False)
    assert decision.action == "allow"
    assert decision.count == 1


def test_freshness_safe_new_turn_resets_success_streak_and_halt():
    controller = _fresh_controller(no_progress_block_after=2)
    args = {"path": "same.txt"}

    for _ in range(2):
        assert controller.before_call("read_file", args).action == "allow"
        decision = controller.after_call("read_file", args, "same", failed=False)
    assert decision.action == "halt"

    controller.reset_for_turn()
    assert controller.before_call("read_file", args).action == "allow"
    fresh = controller.after_call("read_file", args, "same", failed=False)
    assert fresh.action == "allow"
    assert fresh.count == 1


def test_freshness_safe_failure_resets_matching_success_streak():
    controller = _fresh_controller(no_progress_block_after=3, same_tool_failure_halt_after=99)
    args = {"path": "same.txt"}

    for _ in range(2):
        assert controller.before_call("read_file", args).action == "allow"
        assert controller.after_call("read_file", args, "same", failed=False).action == "allow"

    assert controller.before_call("read_file", args).action == "allow"
    assert controller.after_call("read_file", args, '{"error":"transient"}', failed=True).action == "allow"

    assert controller.before_call("read_file", args).action == "allow"
    after_failure = controller.after_call("read_file", args, "same", failed=False)
    assert after_failure.action == "allow"
    assert after_failure.count == 1
    assert controller.halt_decision is None


def test_freshness_safe_process_poll_classification_uses_action_only():
    controller = _fresh_controller(poll_no_progress_block_after=2, no_progress_block_after=2)

    # A mutating process action with a misleading mode must not be counted as a read-only poll.
    mutating_args = {"action": "kill", "mode": "poll", "session_id": "job-1"}
    for _ in range(3):
        assert controller.before_call("process", mutating_args).action == "allow"
        assert controller.after_call("process", mutating_args, "same", failed=False).action == "allow"
    assert controller.halt_decision is None

    # An actual process poll still uses the separate poll budget.
    poll_args = {"action": "poll", "session_id": "job-1"}
    assert controller.before_call("process", poll_args).action == "allow"
    assert controller.after_call("process", poll_args, "same", failed=False).action == "allow"
    assert controller.before_call("process", poll_args).action == "allow"
    halted = controller.after_call("process", poll_args, "same", failed=False)
    assert halted.action == "halt"
    assert halted.code == "poll_repeated_success_halt"


def test_freshness_safe_disabled_preserves_process_poll_success_as_mutating_untracked():
    controller = ToolCallGuardrailController(
        ToolCallGuardrailConfig(
            warnings_enabled=True,
            hard_stop_enabled=True,
            freshness_safe_reads=False,
            no_progress_warn_after=2,
            no_progress_block_after=2,
            poll_no_progress_block_after=2,
        )
    )
    args = {"action": "poll", "session_id": "job-1"}

    for _ in range(4):
        assert controller.before_call("process", args).action == "allow"
        decision = controller.after_call("process", args, "same", failed=False)
        assert decision.action == "allow"
        assert decision.code == "allow"
    assert controller.halt_decision is None


def test_freshness_safe_process_status_poll_uses_separate_budget_and_progress_resets():
    controller = _fresh_controller(poll_no_progress_block_after=12)
    args = {"action": "poll", "session_id": "job-1"}

    for _ in range(11):
        assert controller.before_call("process", args).action == "allow"
        assert controller.after_call("process", args, '{"status":"running","pct":10}', failed=False).action == "allow"

    assert controller.before_call("process", args).action == "allow"
    progress = controller.after_call("process", args, '{"status":"running","pct":20}', failed=False)
    assert progress.action == "allow"
    assert progress.count == 1

    for _ in range(10):
        assert controller.before_call("process", args).action == "allow"
        assert controller.after_call("process", args, '{"status":"running","pct":20}', failed=False).action == "allow"
    assert controller.before_call("process", args).action == "allow"
    halted = controller.after_call("process", args, '{"status":"running","pct":20}', failed=False)
    assert halted.action == "halt"
    assert halted.code == "poll_repeated_success_halt"


def test_freshness_safe_dynamic_browser_observation_is_not_stopped():
    controller = _fresh_controller(no_progress_block_after=2)
    args = {"selector": "body"}

    for _ in range(5):
        assert controller.before_call("browser_snapshot", args).action == "allow"
        decision = controller.after_call("browser_snapshot", args, "same dom", failed=False)
        assert decision.action == "allow"
    assert controller.halt_decision is None
