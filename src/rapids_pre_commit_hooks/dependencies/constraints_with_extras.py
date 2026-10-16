# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import contextlib
from dataclasses import dataclass, field
from typing import Any, Optional, TYPE_CHECKING

import yaml
from packaging.requirements import InvalidRequirement, Requirement

from ..utils.dependencies_yaml import Handler
from ..utils.yaml import (
    Anchor,
    is_reference_anchor,
    preserve_anchor,
    quote_string,
)

if TYPE_CHECKING:
    from collections.abc import Generator

    from ..lint import Linter


class ConstraintsWithExtrasHandler(Handler):
    @dataclass
    class ItemContext:
        output_types: "dict[str, yaml.Node]" = field(default_factory=dict)
        packages: (
            "list[tuple[yaml.ScalarNode, Requirement, Optional[Anchor]]]"
        ) = field(default_factory=list)

    @dataclass
    class PackagesContext:
        parent_context: "ConstraintsWithExtrasHandler.ItemContext"
        is_reference: bool

    def __init__(self, linter: "Linter") -> None:
        self.linter = linter

    @contextlib.contextmanager
    def handle_common_item(
        self,
        common_context: Any,  # noqa: ARG002
        item: "yaml.Node",  # noqa: ARG002
    ) -> "Generator[ConstraintsWithExtrasHandler.ItemContext]":
        context = self.ItemContext()
        yield context

        if "constraints" not in context.output_types:
            return
        for node, req, anchor in context.packages:
            span = (node.start_mark.index, node.end_mark.index)
            warning = self.linter.add_warning(
                span,
                f'package "{req.name}" has extras in a constraints '
                "dependency set",
            )
            constraints_node = context.output_types["constraints"]
            warning.add_note(
                (
                    constraints_node.start_mark.index,
                    constraints_node.end_mark.index,
                ),
                "the constraints output type does not support extras",
            )
            for output_type, output_node in context.output_types.items():
                if output_type != "constraints":
                    warning.add_note(
                        (
                            output_node.start_mark.index,
                            output_node.end_mark.index,
                        ),
                        f"separate the constraints and {output_type} "
                        "dependency lists to preserve extras",
                    )
            if context.output_types.keys() == {"constraints"} and not (
                is_reference_anchor(anchor)
            ):
                req.extras.clear()
                warning.add_replacement(
                    span,
                    preserve_anchor(
                        quote_string(str(req), node.style), anchor
                    ),
                )

    def handle_specific_item(
        self,
        specific_context: Any,
        item: "yaml.Node",
    ) -> (
        "contextlib.AbstractContextManager["
        "ConstraintsWithExtrasHandler.ItemContext]"
    ):
        return self.handle_common_item(specific_context, item)

    def handle_output_type(
        self,
        output_types_context: "ConstraintsWithExtrasHandler.ItemContext",
        item: "yaml.Node",
    ) -> None:
        output_types_context.output_types[item.value] = item

    def handle_packages(
        self,
        common_or_matrices_item_context: (
            "ConstraintsWithExtrasHandler.ItemContext"
        ),
        anchor: "Optional[Anchor]",
        key: "yaml.Node",  # noqa: ARG002
        value: "yaml.Node",  # noqa: ARG002
    ) -> (
        "contextlib.nullcontext[ConstraintsWithExtrasHandler.PackagesContext]"
    ):
        return contextlib.nullcontext(
            self.PackagesContext(
                common_or_matrices_item_context, is_reference_anchor(anchor)
            )
        )

    def handle_package(
        self,
        packages_context: "ConstraintsWithExtrasHandler.PackagesContext",
        anchor: "Optional[Anchor]",
        item: "yaml.ScalarNode",
    ) -> None:
        if packages_context.is_reference:
            return
        try:
            req = Requirement(item.value)
        except InvalidRequirement:
            return
        if req.extras:
            packages_context.parent_context.packages.append(
                (item, req, anchor)
            )
