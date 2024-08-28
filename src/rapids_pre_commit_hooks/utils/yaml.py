# Copyright (c) 2024, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import yaml


def node_has_type(node: "yaml.Node", tag_type: str) -> bool:
    return node.tag == f"tag:yaml.org,2002:{tag_type}"


def get_indent_before_node(content: str, node: "yaml.Node") -> str:
    before_node = content[: node.start_mark.index]
    last_newline = before_node.rfind("\n")
    indent = before_node[last_newline + 1 :]
    assert all(c == " " for c in indent)
    return indent


class AnchorPreservingLoader(yaml.SafeLoader):
    """A SafeLoader that preserves the anchors for later reference. The anchors can
    be found in the document_anchors member, which is a list of dictionaries, one
    dictionary for each parsed document.
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
