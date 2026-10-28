# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable


_AGENT_TESTS = []


def _agent_test(func: "Callable[[], bool]") -> "Callable[[], bool]":
    _AGENT_TESTS.append(func)
    return func


@_agent_test
def is_ai_agent() -> bool:
    ai_agent = os.getenv("AI_AGENT")
    return ai_agent is not None and not re.search(
        r"^(?:|0|false|no|off)$", ai_agent
    )


@_agent_test
def is_agent() -> bool:
    return os.getenv("AGENT") is not None


@_agent_test
def is_codex() -> bool:
    return os.getenv("CODEX_CI") == "1"


def is_agent_environment() -> bool:
    return any(f() for f in _AGENT_TESTS)
