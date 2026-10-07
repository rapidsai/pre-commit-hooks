# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import contextlib

import pytest

from rapids_pre_commit_hooks.lint import LintWarning, Note, Replacement
from rapids_pre_commit_hooks.utils.yaml import load_with_anchors
from rapids_pre_commit_hooks_test_utils import (
    ParseError,
    ParseWarning,
    find_yaml_node_for_span,
    parse_named_spans,
    zip_expected_warnings,
)


@pytest.mark.parametrize(
    ["content", "root_type", "expected_content", "expected_spans", "context"],
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
            id="no-spans",
        ),
        pytest.param(
            "+ Hello\n+ world!\n",
            dict,
            "Hello\nworld!\n",
            {},
            contextlib.nullcontext(),
            id="no-spans-empty-last-line",
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
            id="no-spans-empty-span-line",
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
            id="no-spans-no-newline",
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
            id="no-spans-multiple-no-newlines",
        ),
        pytest.param(
            """\
            + Hello
            :  ^span1
            """,
            dict,
            "Hello\n",
            {
                "span1": (1, 1),
            },
            contextlib.nullcontext(),
            id="single-empty-span",
        ),
        pytest.param(
            """\
            > Hello
            :  ^span1
            """,
            dict,
            "Hello",
            {
                "span1": (1, 1),
            },
            contextlib.nullcontext(),
            id="single-empty-span-no-newline",
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
            id="single-empty-span-at-end",
        ),
        pytest.param(
            """\
            + Hello
            :  ^span1
            :   ^span2
            """,
            dict,
            "Hello\n",
            {
                "span1": (1, 1),
                "span2": (2, 2),
            },
            contextlib.nullcontext(),
            id="multiple-empty-spans",
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
            id="multiple-empty-spans-one-line",
        ),
        pytest.param(
            """\
            + Hello
            :  ~~span1
            """,
            dict,
            "Hello\n",
            {
                "span1": (1, 3),
            },
            contextlib.nullcontext(),
            id="single-nonempty-span",
        ),
        pytest.param(
            """\
            + Hello
            :  >large_span
            + world
            + again
            : !large_span
            """,
            dict,
            "Hello\nworld\nagain\n",
            {
                "large_span": (1, 12),
            },
            contextlib.nullcontext(),
            id="large-span",
        ),
        pytest.param(
            """\
            + Hello
            :  ~~span1  # This is the first span
            """,
            dict,
            "Hello\n",
            {
                "span1": (1, 3),
            },
            contextlib.nullcontext(),
            id="comment",
        ),
        pytest.param(
            """\
            + Hello
            : ~s#~s
            """,
            dict,
            "Hello\n",
            {
                "s": (0, 1),
            },
            contextlib.nullcontext(),
            id="comment-with-span",
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
            :  ~~~~span1
            """,
            dict,
            "Hello\n",
            {
                "span1": (1, 5),
            },
            contextlib.nullcontext(),
            id="single-nonempty-span-to-end-of-line",
        ),
        pytest.param(
            """\
            + Hello
            :  ~~~~~span1
            """,
            dict,
            "Hello\n",
            {
                "span1": (1, 6),
            },
            contextlib.nullcontext(),
            id="single-line-ending-span",
        ),
        pytest.param(
            """\
            + Hello
            :  ~~~~~span1
            + world!
            : ~~span1
            """,
            dict,
            "Hello\nworld!\n",
            {
                "span1": (1, 8),
            },
            contextlib.nullcontext(),
            id="single-multiline-span",
        ),
        pytest.param(
            """\
            + Hello
            :  ~~~~~span1
            :    ~~~span2
            + world!
            : ~~span1
            : ~span2
            """,
            dict,
            "Hello\nworld!\n",
            {
                "span1": (1, 8),
                "span2": (3, 7),
            },
            contextlib.nullcontext(),
            id="multiple-multiline-spans",
        ),
        pytest.param(
            """\
            + Hello
            : ~~span1
            :   ~~span1
            """,
            dict,
            "Hello\n",
            {
                "span1": (0, 4),
            },
            contextlib.nullcontext(),
            id="joined-span-forward",
        ),
        pytest.param(
            """\
            + Hello
            :   ~~span1
            : ~~span1
            """,
            dict,
            "Hello\n",
            {
                "span1": (0, 4),
            },
            contextlib.nullcontext(),
            id="joined-span-reverse",
        ),
        pytest.param(
            """\
            + Hello
            :   ~~~~span1
            + world
            : ~~span1
            """,
            dict,
            "Hello\nworld\n",
            {
                "span1": (2, 8),
            },
            contextlib.nullcontext(),
            id="joined-span-2-lines",
        ),
        pytest.param(
            """\
            + Hello
            :   ~~~~span1
            + world
            : ~~~~~~span1
            + !
            : ~span1
            """,
            dict,
            "Hello\nworld\n!\n",
            {
                "span1": (2, 13),
            },
            pytest.warns(
                ParseWarning,
                match=r'^Span "span1" spans 3 lines, consider using '
                r">/! notation instead$",
            ),
            id="joined-span-3-lines",
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
            : >hello
            :      !hello
            """,
            dict,
            "Hello\n",
            {"hello": (0, 5)},
            pytest.warns(
                ParseWarning,
                match=(
                    r'^Large span "hello" is on a single line, '
                    r"consider using ~ notation instead$"
                ),
            ),
            id="single-line-large-span",
        ),
        pytest.param(
            """\
            + Hello
            :  ~~~~span1
            + world!
            : ~span1
            """,
            dict,
            None,
            None,
            pytest.raises(
                ParseError,
                match=r'^Attempted to create non-contiguous span "span1"$',
            ),
            id="broken-multiline-span-first",
        ),
        pytest.param(
            """\
            + Hello
            :  ~~~~~span1
            + world!
            :  ~span1
            """,
            dict,
            None,
            None,
            pytest.raises(
                ParseError,
                match=r'^Attempted to create non-contiguous span "span1"$',
            ),
            id="broken-multiline-span-second",
        ),
        pytest.param(
            """\
            + Hello
            : ~~s
            :  ~~s
            """,
            dict,
            None,
            None,
            pytest.raises(
                ParseError,
                match=r'^Attempted to create non-contiguous span "s"$',
            ),
            id="overlapping-span",
        ),
        pytest.param(
            """\
            + Hello
            :  ~~~~~~span1
            """,
            dict,
            None,
            None,
            pytest.raises(
                ParseError,
                match=r'^End of span "span1" overruns previous line$',
            ),
            id="past-line-end",
        ),
        pytest.param(
            """\
            + Hello
            : a ~span1
            """,
            dict,
            None,
            None,
            pytest.raises(
                ParseError,
                match=r'^Invalid directive line character: "a"$',
            ),
            id="invalid-before",
        ),
        pytest.param(
            """\
            + Hello
            :   ~span1 a
            """,
            dict,
            None,
            None,
            pytest.raises(
                ParseError,
                match=r'^Invalid directive line character: "a"$',
            ),
            id="invalid-after",
        ),
        pytest.param(
            """\
            + Hello
            @   ~span1
            """,
            dict,
            None,
            None,
            pytest.raises(
                ParseError,
                match=r'^Invalid line start: "@ "$',
            ),
            id="invalid-first-character",
        ),
        pytest.param(
            """\
            +Hello
            """,
            dict,
            None,
            None,
            pytest.raises(
                ParseError,
                match=r'^Invalid line start: "\+H"$',
            ),
            id="content-missing-space",
        ),
        pytest.param(
            """\
            :^a
            """,
            dict,
            None,
            None,
            pytest.raises(
                ParseError,
                match=r'^Invalid line start: ":\^"$',
            ),
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
            pytest.raises(
                ParseError,
                match=r'^Path "0" is a span, but attempted to access "0\.a"$',
            ),
            id="overwrite-span-with-dict",
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
            pytest.raises(
                ParseError,
                match=r'^Path "0" is not a span$',
            ),
            id="overwrite-dict-with-span",
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
            pytest.raises(
                ParseError,
                match=r'^Path "0" is not a list, but got an integer key 0$',
            ),
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
            pytest.raises(
                ParseError,
                match=r'^Path "0" is not a dict, but got a string key "a"$',
            ),
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
            pytest.raises(
                ParseError,
                match=(
                    r'^List "<root>" is missing items at the following '
                    r"indices: 0$"
                ),
            ),
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
            pytest.raises(
                ParseError,
                match=r"^Expected root type to be dict, got list$",
            ),
            id="wrong-root-type",
        ),
        pytest.param(
            """\
            : ~invalid
            """,
            dict,
            None,
            None,
            pytest.raises(
                ParseError,
                match=r'^End of span "invalid" overruns previous line$',
            ),
            id="span-on-no-content",
        ),
        pytest.param(
            """\
            > Hello
            :      ~invalid
            """,
            dict,
            None,
            None,
            pytest.raises(
                ParseError,
                match=r'^End of span "invalid" overruns previous line$',
            ),
            id="newline-on-no-newline",
        ),
        pytest.param(
            """\
            + Hello
            : >s
            :   >s
            :    !s
            """,
            dict,
            None,
            None,
            pytest.raises(
                ParseError,
                match=r'^Large span "s" already in progress$',
            ),
            id="duplicate-large-span",
        ),
        pytest.param(
            """\
            + Hello
            : >s
            + world
            :  !s
            :   !s
            """,
            dict,
            None,
            None,
            pytest.raises(
                ParseError,
                match=r'^Large span "s" not started yet$',
            ),
            id="double-terminate-large-span",
        ),
        pytest.param(
            """\
            + Hello
            : >s
            """,
            dict,
            None,
            None,
            pytest.raises(
                ParseError,
                match=r'^Unfinished large spans: "s"$',
            ),
            id="unterminated-large-span",
        ),
    ],
)
def test_parse_named_spans(
    content, root_type, expected_content, expected_spans, context
):
    with context:
        content, spans = parse_named_spans(content, root_type)
        assert content == expected_content
        assert spans == expected_spans


@pytest.mark.parametrize(
    ["content", "warnings", "expected_warnings"],
    [
        pytest.param(
            """\
            + This is a warning
            : ~~~~0.warning
            :      ~~0.notes.0
            :         ~0.notes.1
            :                  ^0.replacements.0
            : ~~~~0.replacements.1
            :     ^1.warning
            """,
            [
                {
                    "warning": "First warning",
                    "notes": [
                        "First note",
                        "Second note",
                    ],
                    "replacements": [
                        "!",
                        "THIS",
                    ],
                },
                {
                    "warning": "Second warning",
                },
            ],
            [
                LintWarning(
                    (0, 4),
                    "First warning",
                    notes=[
                        Note((5, 7), "First note"),
                        Note((8, 9), "Second note"),
                    ],
                    replacements=[
                        Replacement((17, 17), "!"),
                        Replacement((0, 4), "THIS"),
                    ],
                ),
                LintWarning(
                    (4, 4),
                    "Second warning",
                    notes=[],
                    replacements=[],
                ),
            ],
        ),
    ],
)
def test_zip_expected_warnings(content, warnings, expected_warnings):
    content, spans = parse_named_spans(content, list)
    assert zip_expected_warnings(spans, warnings) == expected_warnings


@pytest.mark.parametrize(
    ["content", "node_lambda"],
    [
        pytest.param(
            """\
            + root_node
            : ~~~~~~~~~node
            """,
            lambda root: root,
            id="basic-string",
        ),
        pytest.param(
            """\
            + 12345
            : ~~~~~node
            """,
            lambda root: root,
            id="basic-number",
        ),
        pytest.param(
            """\
            + null
            : ~~~~node
            """,
            lambda root: root,
            id="basic-null",
        ),
        pytest.param(
            """\
            + key1: value1
            : >node
            + key2: value2
            :              !node
            """,
            lambda root: root,
            id="map-root",
        ),
        pytest.param(
            """\
            + key1: value1
            : ~~~~node
            + key2: value2
            """,
            lambda root: root.value[0][0],
            id="map-key-1",
        ),
        pytest.param(
            """\
            + key1: value1
            :       ~~~~~~node
            + key2: value2
            """,
            lambda root: root.value[0][1],
            id="map-value-1",
        ),
        pytest.param(
            """\
            + key1: value1
            + key2: value2
            : ~~~~node
            """,
            lambda root: root.value[1][0],
            id="map-key-2",
        ),
        pytest.param(
            """\
            + key1: value1
            + key2: value2
            :       ~~~~~~node
            """,
            lambda root: root.value[1][1],
            id="map-value-2",
        ),
        pytest.param(
            """\
            + - item1
            : >node
            + - item2
            :         !node
            """,
            lambda root: root,
            id="seq-root",
        ),
        pytest.param(
            """\
            + - item1
            :   ~~~~~node
            + - item2
            """,
            lambda root: root.value[0],
            id="seq-item-1",
        ),
        pytest.param(
            """\
            + - item1
            + - item2
            :   ~~~~~node
            """,
            lambda root: root.value[1],
            id="seq-item-2",
        ),
        pytest.param(
            """\
            + root_node
            :  ~~~node
            """,
            lambda _root: None,
            id="no-node",
        ),
    ],
)
def test_find_yaml_node_for_span(content, node_lambda):
    content, spans = parse_named_spans(content)
    root, _ = load_with_anchors(content)

    assert find_yaml_node_for_span(root, spans["node"]) == node_lambda(root)
