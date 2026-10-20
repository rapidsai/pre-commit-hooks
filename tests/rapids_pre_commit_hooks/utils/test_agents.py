# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from unittest.mock import Mock, patch

from rapids_pre_commit_hooks.utils import agents
from rapids_pre_commit_hooks_test_utils import patch_environment


def test_agent_test_decorator():
    with patch(
        "rapids_pre_commit_hooks.utils.agents._AGENT_TESTS", []
    ) as mock_tests:

        @agents._agent_test
        def test():
            return True

        assert test()
        assert mock_tests == [test]


@pytest.mark.parametrize(
    ["env", "result"],
    [
        pytest.param(None, False, id="not-present"),
        pytest.param("", False, id="empty"),
        *(
            pytest.param(value, False, id=value)
            for value in ["0", "false", "no", "off"]
        ),
        *(
            pytest.param(value, True, id=value)
            for value in ["claude", "codex", "grok", "cursor", "gemini"]
        ),
    ],
)
def test_is_ai_agent(env, result):
    with patch_environment({"AI_AGENT": env}):
        assert agents.is_ai_agent() == result


@pytest.mark.parametrize(
    ["env", "result"],
    [
        pytest.param(None, False, id="not-present"),
        pytest.param("", True, id="empty"),
        *(
            pytest.param(value, True, id=value)
            for value in ["1", "true", "claude", "codex"]
        ),
    ],
)
def test_is_agent(env, result):
    with patch_environment({"AGENT": env}):
        assert agents.is_agent() == result


@pytest.mark.parametrize(
    ["env", "result"],
    [
        pytest.param(None, False, id="not-present"),
        pytest.param("", False, id="empty"),
        pytest.param("0", False, id="0"),
        pytest.param("1", True, id="1"),
    ],
)
def test_is_codex(env, result):
    with patch_environment({"CODEX_CI": env}):
        assert agents.is_codex() == result


@pytest.mark.parametrize(
    ["values", "called", "result"],
    [
        pytest.param([False, False], [True, True], False, id="none"),
        pytest.param([True, False], [True, False], True, id="first"),
        pytest.param([False, True], [True, True], True, id="second"),
        pytest.param([True, True], [True, False], True, id="both"),
    ],
)
def test_is_agent_environment(values, called, result):
    with patch(
        "rapids_pre_commit_hooks.utils.agents._AGENT_TESTS",
        [Mock(return_value=v) for v in values],
    ) as tests:
        assert agents.is_agent_environment() == result
        assert [t.called for t in tests] == called
