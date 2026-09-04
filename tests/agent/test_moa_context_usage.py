"""MoA context accounting regression tests."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from agent.conversation_loop import _context_usage_dict
from agent.usage_pricing import CanonicalUsage
from gateway.runtime_footer import format_runtime_footer
from run_agent import AIAgent


def test_context_usage_prefers_acting_moa_aggregator_over_reported_fanout():
    aggregator = CanonicalUsage(
        input_tokens=68_000,
        output_tokens=900,
        cache_read_tokens=2_000,
        cache_write_tokens=100,
    )
    reported = aggregator + CanonicalUsage(
        input_tokens=40_000,
        output_tokens=600,
        cache_read_tokens=1_000,
    )

    usage = _context_usage_dict(aggregator_usage=aggregator, reported_usage=reported)

    assert usage == {
        "prompt_tokens": 70_100,
        "completion_tokens": 900,
        "total_tokens": 71_000,
        "input_tokens": 68_000,
        "output_tokens": 900,
        "cache_read_tokens": 2_000,
        "cache_write_tokens": 100,
        "reasoning_tokens": 0,
    }


def test_moa_advisor_usage_stays_out_of_compressor_finalizer_and_footer():
    """Exercise MoA accounting through the real loop, finalizer, and footer."""
    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://example.invalid/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )

    agent._cached_system_prompt = "You are helpful."
    agent._use_prompt_caching = False
    agent.compression_enabled = False
    agent.save_trajectories = False
    agent.provider = "moa"
    agent.model = "closed"

    details = SimpleNamespace(cached_tokens=2_000, cache_write_tokens=100)
    usage = SimpleNamespace(
        prompt_tokens=70_100,
        completion_tokens=900,
        prompt_tokens_details=details,
    )
    message = SimpleNamespace(
        content="Final answer",
        tool_calls=[],
        reasoning=None,
        reasoning_content=None,
        reasoning_details=None,
    )
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason="stop")],
        model="closed",
        usage=usage,
    )

    client = MagicMock()
    client.chat.completions.create.return_value = response
    client.consume_reference_usage.return_value = (
        CanonicalUsage(input_tokens=40_000, output_tokens=600, cache_read_tokens=1_000),
        0.0,
    )
    client.last_aggregator_slot = {
        "model": "gpt-5.6-sol",
        "provider": "openai",
        "base_url": "",
    }
    agent.client = client

    with (
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("hello")

    assert result["completed"] is True
    assert result["input_tokens"] == 108_000
    assert result["output_tokens"] == 1_500
    assert result["cache_read_tokens"] == 3_000
    assert result["last_prompt_tokens"] == 70_100

    footer = format_runtime_footer(
        model=result["model"],
        context_tokens=result["last_prompt_tokens"],
        context_length=100_000,
        cwd="",
        fields=["context_pct"],
    )
    assert footer == "70%"
