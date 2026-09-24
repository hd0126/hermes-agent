"""Exercise the existing conversation loop, not a second tool dispatcher."""

import json
from pathlib import Path
from unittest.mock import patch

from tests.run_agent.test_tool_call_guardrail_runtime import (
    _hard_stop_config,
    _make_agent,
    _mock_response,
    _mock_tool_call,
)


def _responses(tool, args, count):
    return [
        _mock_response(content='', finish_reason='tool_calls', tool_calls=[
            _mock_tool_call(tool, json.dumps(args), f'repeat-{i}')
        ]) for i in range(count)
    ]


def _config():
    config = _hard_stop_config(freshness_safe_reads=True)
    config['tool_loop_guardrails']['hard_stop_after']['idempotent_no_progress'] = 3
    return config


def test_repeated_skill_read_halts_real_loop_before_fourth_model_request():
    agent = _make_agent('skill_view', config=_config())
    agent.client.chat.completions.create.side_effect = _responses('skill_view', {'name': 'example'}, 8)
    with (
        patch('run_agent.handle_function_call', return_value=json.dumps({'content': 'same rules'})) as dispatch,
        patch.object(agent, '_persist_session'),
        patch.object(agent, '_save_trajectory'),
        patch.object(agent, '_cleanup_task_resources'),
    ):
        result = agent.run_conversation('Read the example rules.')
    assert dispatch.call_count == 3
    assert agent.client.chat.completions.create.call_count == 3
    assert result['turn_exit_reason'] == 'guardrail_halt'
    assert result['final_response']
    assert len([m for m in result['messages'] if m.get('role') == 'tool']) == 3


def test_changed_file_is_re_read_before_any_no_progress_decision(tmp_path):
    source = tmp_path / 'source.txt'
    source.write_text('first')
    agent = _make_agent('read_file', config=_config())
    args = {'path': str(source)}
    agent.client.chat.completions.create.side_effect = (
        _responses('read_file', args, 3) + [_mock_response(content='done')]
    )
    observed = []

    def read_source(*unused_args, **unused_kwargs):
        if len(observed) == 2:
            source.write_text('changed by another actor')
        text = Path(args['path']).read_text()
        observed.append(text)
        return json.dumps({'content': text})

    with (
        patch('run_agent.handle_function_call', side_effect=read_source),
        patch.object(agent, '_persist_session'),
        patch.object(agent, '_save_trajectory'),
        patch.object(agent, '_cleanup_task_resources'),
    ):
        result = agent.run_conversation('Inspect the file.')
    assert observed == ['first', 'first', 'changed by another actor']
    assert result['final_response'] == 'done'
    assert result['turn_exit_reason'].startswith('text_response')
