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
