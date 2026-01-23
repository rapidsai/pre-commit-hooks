# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import pytest
import tomlkit

from rapids_pre_commit_hooks.utils.toml import find_value_location
from rapids_pre_commit_hooks_test_utils import parse_named_ranges


@pytest.mark.parametrize(
    ["key", "append"],
    [
        pytest.param(
            ("table", "key1"),
            False,
            id="string-value",
        ),
        pytest.param(
            ("table", "key2"),
            False,
            id="int-value",
        ),
        pytest.param(
            ("table", "key3"),
            False,
            id="subtable",
        ),
        pytest.param(
            ("table", "key3", "nested"),
            False,
            id="subtable-nested",
        ),
        pytest.param(
            ("table", "key4"),
            False,
            id="inline-comments",
        ),
        pytest.param(
            ("table",),
            False,
            id="table",
        ),
        pytest.param(
            ("table",),
            True,
            id="append",
        ),
        pytest.param(
            ("table2",),
            False,
            id="table2",
        ),
    ],
)
def test_find_value_location(key, append):
    content, positions = parse_named_ranges(
        """\
        + [table]
        + key1 = "value"
        : >table._value
        :        ~~~~~~~table.key1._value
        + key2 = 42
        :        ~~table.key2._value
        + key3 = { nested = "value" }
        :        ~~~~~~~~~~~~~~~~~~~~table.key3._value
        :                   ~~~~~~~table.key3.nested._value
        + key4 = "beep-boop" # and a trailing comment
        :        ~~~~~~~~~~~table.key4._value
        +
        : ^table._append
        + [table2]
        : !table._value
        + key = "value"
        : ~~~~~~~~~~~~~~table2._value
        """
    )
    parsed_doc = tomlkit.loads(content)
    loc = positions
    for component in key:
        loc = loc[component]
    loc = loc["_append" if append else "_value"]
    assert find_value_location(parsed_doc, key, append=append) == loc
    assert parsed_doc.as_string() == content
