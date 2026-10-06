# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import dataclasses
from enum import Enum
from typing import Optional

import yaml


class AnchorPreservingLoader(yaml.SafeLoader):
    """A SafeLoader that preserves the anchors for later reference. The anchors
    can be found in the document_anchors member, which is a list of
    dictionaries, one dictionary for each parsed document.
    """

    def __init__(self, stream) -> None:
        super().__init__(stream)
        self.document_anchors: list[dict[str, yaml.Node]] = []

    def compose_document(self) -> "yaml.Node":
        # Drop the DOCUMENT-START event.
        self.get_event()

        # Compose the root node.
        node = self.compose_node(None, None)  # type: ignore[arg-type]

        # Drop the DOCUMENT-END event.
        self.get_event()

        self.document_anchors.append(self.anchors)
        self.anchors = {}
        assert node is not None
        return node


class AnchorType(Enum):
    DEFINITION = 0
    REFERENCE = 1


@dataclasses.dataclass
class Anchor:
    anchor_type: AnchorType
    anchor_name: str


def node_has_type(node: "yaml.Node", tag_type: str) -> bool:
    return node.tag == f"tag:yaml.org,2002:{tag_type}"


def check_and_mark_anchor(
    anchors: "dict[str, yaml.Node]", used_anchors: set[str], node: "yaml.Node"
) -> "Optional[Anchor]":
    for key, value in anchors.items():
        if value == node:
            anchor = key
            break
    else:
        anchor = None
    if anchor in used_anchors:
        return Anchor(AnchorType.REFERENCE, anchor)
    if anchor is not None:
        used_anchors.add(anchor)
        return Anchor(AnchorType.DEFINITION, anchor)
    return None


def is_reference_anchor(anchor: "Optional[Anchor]") -> bool:
    return anchor is not None and anchor.anchor_type == AnchorType.REFERENCE


def load_with_anchors(stream) -> "tuple[yaml.Node, dict[str, yaml.Node]]":
    loader = AnchorPreservingLoader(stream)
    try:
        root = loader.get_single_node()
        assert root is not None
        return root, dict(loader.document_anchors[0])
    finally:
        loader.dispose()
