# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock

import pytest

from rapids_pre_commit_hooks.utils.yaml import (
    Anchor,
    AnchorType,
    check_and_mark_anchor,
    is_reference_anchor,
    load_with_anchors,
    preserve_anchor,
    quote_string,
)
from rapids_pre_commit_hooks_test_utils import (
    find_yaml_node_for_span,
    parse_named_spans,
)


def test_load_with_anchors():
    content, spans = parse_named_spans(
        """\
        + - &a A
        :   ~~~~anchor
        + - *a
        """
    )
    root, anchors = load_with_anchors(content)
    assert anchors == {
        "a": find_yaml_node_for_span(root, spans["anchor"]),
    }


@pytest.mark.parametrize(
    [
        "used_anchors_before",
        "node_index",
        "anchor",
        "used_anchors_after",
    ],
    [
        (
            set(),
            0,
            Anchor(AnchorType.DEFINITION, "anchor1"),
            {"anchor1"},
        ),
        (
            {"anchor1"},
            1,
            Anchor(AnchorType.DEFINITION, "anchor2"),
            {"anchor1", "anchor2"},
        ),
        (
            set(),
            2,
            None,
            set(),
        ),
        (
            {"anchor1", "anchor2"},
            0,
            Anchor(AnchorType.REFERENCE, "anchor1"),
            {"anchor1", "anchor2"},
        ),
        (
            {"anchor1", "anchor2"},
            1,
            Anchor(AnchorType.REFERENCE, "anchor2"),
            {"anchor1", "anchor2"},
        ),
    ],
)
def test_check_and_mark_anchor(
    used_anchors_before,
    node_index,
    anchor,
    used_anchors_after,
):
    NODES = [Mock() for _ in range(3)]
    ANCHORS = {
        "anchor1": NODES[0],
        "anchor2": NODES[1],
    }
    used_anchors = set(used_anchors_before)
    actual_anchor = check_and_mark_anchor(
        ANCHORS, used_anchors, NODES[node_index]
    )
    assert actual_anchor == anchor
    assert used_anchors == used_anchors_after


@pytest.mark.parametrize(
    ["anchor", "is_ref"],
    [
        pytest.param(
            None,
            False,
            id="none",
        ),
        pytest.param(
            Anchor(AnchorType.DEFINITION, "anchor"),
            False,
            id="definition",
        ),
        pytest.param(
            Anchor(AnchorType.REFERENCE, "anchor"),
            True,
            id="reference",
        ),
    ],
)
def test_is_reference_anchor(anchor, is_ref):
    assert is_reference_anchor(anchor) == is_ref


@pytest.mark.parametrize(
    ["value", "style", "expected"],
    [
        pytest.param("package>=1", None, "package>=1", id="requirement-plain"),
        pytest.param(
            "package>=1", "'", "'package>=1'", id="requirement-single-quoted"
        ),
        pytest.param(
            "package>=1", '"', '"package>=1"', id="requirement-double-quoted"
        ),
        pytest.param(
            "it's a package", None, "it's a package", id="apostrophe-plain"
        ),
        pytest.param(
            "it's a package",
            "'",
            "'it''s a package'",
            id="apostrophe-single-quoted",
        ),
        pytest.param(
            "it's a package",
            '"',
            '"it\'s a package"',
            id="apostrophe-double-quoted",
        ),
        pytest.param(
            'a "quote"', None, 'a "quote"', id="quotation-marks-plain"
        ),
        pytest.param(
            'a "quote"',
            "'",
            "'a \"quote\"'",
            id="quotation-marks-single-quoted",
        ),
        pytest.param(
            'a "quote"',
            '"',
            '"a \\"quote\\""',
            id="quotation-marks-double-quoted",
        ),
        pytest.param("a\\b", None, "a\\b", id="backslash-plain"),
        pytest.param("a\\b", "'", "'a\\b'", id="backslash-single-quoted"),
        pytest.param("a\\b", '"', '"a\\\\b"', id="backslash-double-quoted"),
        pytest.param("café", None, "café", id="unicode-plain"),
        pytest.param("café", "'", "'café'", id="unicode-single-quoted"),
        pytest.param("café", '"', '"café"', id="unicode-double-quoted"),
    ],
)
def test_quote_string(value, style, expected):
    assert quote_string(value, style) == expected


@pytest.mark.parametrize(
    ["value", "anchor", "expected"],
    [
        pytest.param("package", None, "package", id="no-anchor"),
        pytest.param(
            "package",
            Anchor(AnchorType.DEFINITION, "package_anchor"),
            "&package_anchor package",
            id="anchor-definition",
        ),
        pytest.param(
            "'package'",
            Anchor(AnchorType.DEFINITION, "package_anchor"),
            "&package_anchor 'package'",
            id="single-quoted-anchor-definition",
        ),
        pytest.param(
            '"package"',
            Anchor(AnchorType.DEFINITION, "package_anchor"),
            '&package_anchor "package"',
            id="double-quoted-anchor-definition",
        ),
    ],
)
def test_preserve_anchor(value, anchor, expected):
    assert preserve_anchor(value, anchor) == expected
