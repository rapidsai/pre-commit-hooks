# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import contextlib

import pytest

from rapids_pre_commit_hooks_test_utils import ParseError, parse_named_ranges


@pytest.mark.parametrize(
    ["content", "root_type", "expected_content", "expected_ranges", "context"],
    [
        pytest.param(
            "+",
            None,
            "",
            None,
            contextlib.nullcontext(),
            id="empty-content-none",
        ),
        pytest.param(
            "+",
            dict,
            "",
            {},
            contextlib.nullcontext(),
            id="empty-content-dict",
        ),
        pytest.param(
            "+",
            list,
            "",
            [],
            contextlib.nullcontext(),
            id="empty-content-list",
        ),
        pytest.param(
            "+ Hello\n+ world!\n+",
            dict,
            "Hello\nworld!\n",
            {},
            contextlib.nullcontext(),
            id="no-ranges",
        ),
        pytest.param(
            "+ Hello\n+ world!\n",
            dict,
            "Hello\nworld!\n",
            {},
            contextlib.nullcontext(),
            id="no-ranges-empty-last-line",
        ),
        pytest.param(
            """\
            + Hello
            + world!
            :""",
            dict,
            "Hello\nworld!\n",
            {},
            contextlib.nullcontext(),
            id="no-ranges-empty-range-line",
        ),
        pytest.param(
            """\
            + Hello
            > world!
            :""",
            dict,
            "Hello\nworld!",
            {},
            contextlib.nullcontext(),
            id="no-ranges-no-newline",
        ),
        pytest.param(
            """\
            > Hello
            >  world!
            :""",
            dict,
            "Hello world!",
            {},
            contextlib.nullcontext(),
            id="no-ranges-multiple-no-newlines",
        ),
        pytest.param(
            """\
            + Hello
            :  ^group1
            """,
            dict,
            "Hello\n",
            {
                "group1": (1, 1),
            },
            contextlib.nullcontext(),
            id="single-empty-group",
        ),
        pytest.param(
            """\
            > Hello
            :  ^group1
            """,
            dict,
            "Hello",
            {
                "group1": (1, 1),
            },
            contextlib.nullcontext(),
            id="single-empty-group-no-newline",
        ),
        pytest.param(
            """\
            + Hello
            :       ^end
            """,
            dict,
            "Hello\n",
            {
                "end": (6, 6),
            },
            contextlib.nullcontext(),
            id="single-empty-group-at-end",
        ),
        pytest.param(
            """\
            + Hello
            :  ^group1
            :   ^group2
            """,
            dict,
            "Hello\n",
            {
                "group1": (1, 1),
                "group2": (2, 2),
            },
            contextlib.nullcontext(),
            id="multiple-empty-groups",
        ),
        pytest.param(
            """\
            + Hello
            : ^a  ^b
            """,
            dict,
            "Hello\n",
            {
                "a": (0, 0),
                "b": (4, 4),
            },
            contextlib.nullcontext(),
            id="multiple-empty-groups-one-line",
        ),
        pytest.param(
            """\
            + Hello
            :  ~~group1
            """,
            dict,
            "Hello\n",
            {
                "group1": (1, 3),
            },
            contextlib.nullcontext(),
            id="single-nonempty-group",
        ),
        pytest.param(
            """\
            + Hello
            :  >large_group
            + world
            + again
            : !large_group
            """,
            dict,
            "Hello\nworld\nagain\n",
            {
                "large_group": (1, 12),
            },
            contextlib.nullcontext(),
            id="large-group",
        ),
        pytest.param(
            """\
            + Hello
            :  ~~group1  # This is the first group
            """,
            dict,
            "Hello\n",
            {
                "group1": (1, 3),
            },
            contextlib.nullcontext(),
            id="comment",
        ),
        pytest.param(
            """\
            + Hello
            : ~g#~g
            """,
            dict,
            "Hello\n",
            {
                "g": (0, 1),
            },
            contextlib.nullcontext(),
            id="comment-with-range",
        ),
        pytest.param(
            """\
            +
            :
            """,
            dict,
            "\n",
            {},
            contextlib.nullcontext(),
            id="empty-lines",
        ),
        pytest.param(
            """\
            + Hello
            :  ~~~~group1
            """,
            dict,
            "Hello\n",
            {
                "group1": (1, 5),
            },
            contextlib.nullcontext(),
            id="single-nonempty-group-to-end-of-line",
        ),
        pytest.param(
            """\
            + Hello
            :  ~~~~~group1
            """,
            dict,
            "Hello\n",
            {
                "group1": (1, 6),
            },
            contextlib.nullcontext(),
            id="single-line-ending-group",
        ),
        pytest.param(
            """\
            + Hello
            :  ~~~~~group1
            + world!
            : ~~group1
            """,
            dict,
            "Hello\nworld!\n",
            {
                "group1": (1, 8),
            },
            contextlib.nullcontext(),
            id="single-multiline-group",
        ),
        pytest.param(
            """\
            + Hello
            :  ~~~~~group1
            :    ~~~group2
            + world!
            : ~~group1
            : ~group2
            """,
            dict,
            "Hello\nworld!\n",
            {
                "group1": (1, 8),
                "group2": (3, 7),
            },
            contextlib.nullcontext(),
            id="multiple-multiline-groups",
        ),
        pytest.param(
            """\
            + Hello
            : ~~group1
            :   ~~group1
            """,
            dict,
            "Hello\n",
            {
                "group1": (0, 4),
            },
            contextlib.nullcontext(),
            id="joined-group-forward",
        ),
        pytest.param(
            """\
            + Hello
            :   ~~group1
            : ~~group1
            """,
            dict,
            "Hello\n",
            {
                "group1": (0, 4),
            },
            contextlib.nullcontext(),
            id="joined-group-reverse",
        ),
        pytest.param(
            """\
            + Hello
            : ~0 ~1
            """,
            list,
            "Hello\n",
            [(0, 1), (3, 4)],
            contextlib.nullcontext(),
            id="simple-list-forward",
        ),
        pytest.param(
            """\
            + Hello
            : ~1 ~0
            """,
            list,
            "Hello\n",
            [(3, 4), (0, 1)],
            contextlib.nullcontext(),
            id="simple-list-reverse",
        ),
        pytest.param(
            """\
            + Hello
            : ~0.a
            :  ~0.b
            :   ~1.a
            :    ~1.b
            """,
            list,
            "Hello\n",
            [
                {"a": (0, 1), "b": (1, 2)},
                {"a": (2, 3), "b": (3, 4)},
            ],
            contextlib.nullcontext(),
            id="dict-in-list",
        ),
        pytest.param(
            """\
            + Hello
            : ~a.0
            :  ~a.1
            :   ~b.0
            :    ~b.1
            """,
            dict,
            "Hello\n",
            {
                "a": [(0, 1), (1, 2)],
                "b": [(2, 3), (3, 4)],
            },
            contextlib.nullcontext(),
            id="list-in-dict",
        ),
        pytest.param(
            """\
            + Hello
            : ~a.a
            :  ~a.b
            :   ~b.a
            :    ~b.b
            """,
            dict,
            "Hello\n",
            {
                "a": {"a": (0, 1), "b": (1, 2)},
                "b": {"a": (2, 3), "b": (3, 4)},
            },
            contextlib.nullcontext(),
            id="dict-in-dict",
        ),
        pytest.param(
            """\
            + Hello
            : ~0.0
            :  ~0.1
            :   ~1.0
            :    ~1.1
            """,
            list,
            "Hello\n",
            [
                [(0, 1), (1, 2)],
                [(2, 3), (3, 4)],
            ],
            contextlib.nullcontext(),
            id="list-in-list",
        ),
        pytest.param(
            """\
            + Hello
            : ~0.a.1
            :  ~0.a.0
            :   ~1.b.c.0.d
            :    ~2
            """,
            None,
            "Hello\n",
            [
                {"a": [(1, 2), (0, 1)]},
                {"b": {"c": [{"d": (2, 3)}]}},
                (3, 4),
            ],
            contextlib.nullcontext(),
            id="complex",
        ),
        pytest.param(
            """\
            + Hello
            : ~a
            """,
            None,
            "Hello\n",
            {"a": (0, 1)},
            contextlib.nullcontext(),
            id="root-type-none-dict",
        ),
        pytest.param(
            """\
            + Hello
            : ~0
            """,
            None,
            "Hello\n",
            [(0, 1)],
            contextlib.nullcontext(),
            id="root-type-none-list",
        ),
        pytest.param(
            """\
            + Hello
            :  ~~~~group1
            + world!
            : ~group1
            """,
            dict,
            None,
            None,
            pytest.raises(ParseError),
            id="broken-multiline-group-first",
        ),
        pytest.param(
            """\
            + Hello
            :  ~~~~~group1
            + world!
            :  ~group1
            """,
            dict,
            None,
            None,
            pytest.raises(ParseError),
            id="broken-multiline-group-second",
        ),
        pytest.param(
            """\
            + Hello
            : ~~g
            :  ~~g
            """,
            dict,
            None,
            None,
            pytest.raises(ParseError),
            id="overlapping-group",
        ),
        pytest.param(
            """\
            + Hello
            :  ~~~~~~group1
            """,
            dict,
            None,
            None,
            pytest.raises(ParseError),
            id="past-line-end",
        ),
        pytest.param(
            """\
            + Hello
            : a ~group1
            """,
            dict,
            None,
            None,
            pytest.raises(ParseError),
            id="invalid-before",
        ),
        pytest.param(
            """\
            + Hello
            :   ~group1 a
            """,
            dict,
            None,
            None,
            pytest.raises(ParseError),
            id="invalid-after",
        ),
        pytest.param(
            """\
            + Hello
            @   ~group1
            """,
            dict,
            None,
            None,
            pytest.raises(ParseError),
            id="invalid-first-character",
        ),
        pytest.param(
            """\
            +Hello
            """,
            dict,
            None,
            None,
            pytest.raises(ParseError),
            id="content-missing-space",
        ),
        pytest.param(
            """\
            :^a
            """,
            dict,
            None,
            None,
            pytest.raises(ParseError),
            id="directive-missing-space",
        ),
        pytest.param(
            """\
            + Hello
            : ~0
            :  ~0.a
            """,
            list,
            None,
            None,
            pytest.raises(ParseError),
            id="overwrite-range-with-dict",
        ),
        pytest.param(
            """\
            + Hello
            : ~0.a
            :  ~0
            """,
            list,
            None,
            None,
            pytest.raises(ParseError),
            id="overwrite-dict-with-range",
        ),
        pytest.param(
            """\
            + Hello
            : ~0.a
            :  ~0.0
            """,
            list,
            None,
            None,
            pytest.raises(ParseError),
            id="overwrite-dict-with-list",
        ),
        pytest.param(
            """\
            + Hello
            : ~0.0
            :  ~0.a
            """,
            list,
            None,
            None,
            pytest.raises(ParseError),
            id="overwrite-list-with-dict",
        ),
        pytest.param(
            """\
            + Hello
            : ~1
            """,
            list,
            None,
            None,
            pytest.raises(ParseError),
            id="incomplete-list",
        ),
        pytest.param(
            """\
            + Hello
            : ~0
            """,
            dict,
            None,
            None,
            pytest.raises(ParseError),
            id="wrong-root-type",
        ),
        pytest.param(
            """\
            : ~invalid
            """,
            dict,
            None,
            None,
            pytest.raises(ParseError),
            id="group-on-no-content",
        ),
        pytest.param(
            """\
            > Hello
            :      ~invalid
            """,
            dict,
            None,
            None,
            pytest.raises(ParseError),
            id="newline-on-no-newline",
        ),
        pytest.param(
            """\
            + Hello
            : >g
            :   >g
            :    !g
            """,
            dict,
            None,
            None,
            pytest.raises(ParseError),
            id="duplicate-large-group",
        ),
        pytest.param(
            """\
            + Hello
            : >g
            :  !g
            :   !g
            """,
            dict,
            None,
            None,
            pytest.raises(ParseError),
            id="double-terminate-large-group",
        ),
        pytest.param(
            """\
            + Hello
            : >g
            """,
            dict,
            None,
            None,
            pytest.raises(ParseError),
            id="unterminated-large-group",
        ),
    ],
)
def test_parse_named_ranges(
    content, root_type, expected_content, expected_ranges, context
):
    with context:
        content, named_ranges = parse_named_ranges(content, root_type)
        assert content == expected_content
        assert named_ranges == expected_ranges
