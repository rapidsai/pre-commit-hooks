# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import contextlib
from typing import Any, Optional, TYPE_CHECKING

import yaml

from .yaml import (
    Anchor,
    AnchorPreservingLoader,
    check_and_mark_anchor,
    node_has_type,
)

if TYPE_CHECKING:
    from collections.abc import Generator, Iterable


class Handler:
    def handle_root(
        self,
        value: "yaml.Node",  # noqa: ARG002
    ) -> "contextlib.AbstractContextManager[Any]":
        return contextlib.nullcontext()

    def handle_dependencies(
        self,
        root_context: "Any",
        key: "yaml.Node",  # noqa: ARG002
        value: "yaml.Node",  # noqa: ARG002
    ) -> "contextlib.AbstractContextManager[Any]":
        return contextlib.nullcontext(root_context)

    def handle_dependency_set(
        self,
        dependencies_context: "Any",
        key: "yaml.Node",  # noqa: ARG002
        value: "yaml.Node",  # noqa: ARG002
    ) -> "contextlib.AbstractContextManager[Any]":
        return contextlib.nullcontext(dependencies_context)

    def handle_common(
        self,
        dependency_set_context: "Any",
        key: "yaml.Node",  # noqa: ARG002
        value: "yaml.Node",  # noqa: ARG002
    ) -> "contextlib.AbstractContextManager[Any]":
        return contextlib.nullcontext(dependency_set_context)

    def handle_common_item(
        self,
        common_context: "Any",
        item: "yaml.Node",  # noqa: ARG002
    ) -> "contextlib.AbstractContextManager[Any]":
        return contextlib.nullcontext(common_context)

    def handle_output_types(
        self,
        common_item_or_specific_item_context: "Any",
        key: "yaml.Node",  # noqa: ARG002
        value: "yaml.Node",  # noqa: ARG002
    ) -> "contextlib.AbstractContextManager[Any]":
        return contextlib.nullcontext(common_item_or_specific_item_context)

    def handle_output_type(
        self,
        output_types_context: "Any",
        item: "yaml.Node",  # noqa: ARG002
    ) -> None:
        pass

    def handle_specific(
        self,
        dependency_set_context: "Any",
        key: "yaml.Node",  # noqa: ARG002
        value: "yaml.Node",  # noqa: ARG002
    ) -> "contextlib.AbstractContextManager[Any]":
        return contextlib.nullcontext(dependency_set_context)

    def handle_specific_item(
        self,
        specific_context: "Any",
        item: "yaml.Node",  # noqa: ARG002
    ) -> "contextlib.AbstractContextManager[Any]":
        return contextlib.nullcontext(specific_context)

    def handle_matrices(
        self,
        specific_item_context: "Any",
        key: "yaml.Node",  # noqa: ARG002
        value: "yaml.Node",  # noqa: ARG002
    ) -> "contextlib.AbstractContextManager[Any]":
        return contextlib.nullcontext(specific_item_context)

    def handle_matrices_item(
        self,
        matrices_context: "Any",
        item: "yaml.Node",  # noqa: ARG002
    ) -> "contextlib.AbstractContextManager[Any]":
        return contextlib.nullcontext(matrices_context)

    def handle_matrix(
        self,
        matrices_item_context: "Any",
        key: "yaml.Node",  # noqa: ARG002
        value: "yaml.Node",  # noqa: ARG002
    ) -> "contextlib.AbstractContextManager[Any]":
        return contextlib.nullcontext(matrices_item_context)

    def handle_matrix_item(
        self,
        matrix_context: "Any",  # noqa: ARG002
        key: "yaml.Node",  # noqa: ARG002
        value: "yaml.Node",  # noqa: ARG002
    ) -> None:
        pass

    def handle_packages(
        self,
        common_or_matrices_item_context: "Any",
        anchor: "Optional[Anchor]",  # noqa: ARG002
        key: "yaml.Node",  # noqa: ARG002
        value: "yaml.Node",  # noqa: ARG002
    ) -> "contextlib.AbstractContextManager[Any]":
        return contextlib.nullcontext(common_or_matrices_item_context)

    def handle_package(
        self,
        packages_context: "Any",  # noqa: ARG002
        anchor: "Optional[Anchor]",  # noqa: ARG002
        item: "yaml.Node",  # noqa: ARG002
    ) -> None:
        pass


class ChainedHandler(Handler):
    def __init__(self) -> None:
        self.handlers: "list[Handler]" = []

    def add_handler(self, handler: "Handler") -> None:
        self.handlers.append(handler)

    def _handlers_with_context_arg(
        self, parent_context: "Optional[tuple[Any, ...]]"
    ) -> "Iterable[tuple[Handler, Any]]":
        return zip(
            self.handlers,
            [()] * len(self.handlers)
            if parent_context is None
            else map(lambda c: (c,), parent_context),
            strict=True,
        )

    @contextlib.contextmanager
    def _handle_context(
        self,
        hook_name: str,
        parent_context: "Optional[tuple[Any, ...]]",
        *args,
        **kwargs,
    ) -> "Generator[tuple[Any, ...]]":
        with contextlib.ExitStack() as context:
            yield tuple(
                context.enter_context(
                    getattr(handler, hook_name)(
                        *handler_context_arg, *args, **kwargs
                    )
                )
                for handler, handler_context_arg in (
                    self._handlers_with_context_arg(parent_context)
                )
            )

    def _handle_no_context(
        self,
        hook_name: str,
        parent_context: "Optional[tuple[Any, ...]]",
        *args,
        **kwargs,
    ) -> None:
        for handler, handler_context_arg in self._handlers_with_context_arg(
            parent_context
        ):
            getattr(handler, hook_name)(*handler_context_arg, *args, **kwargs)

    def handle_root(
        self, *args, **kwargs
    ) -> "contextlib.AbstractContextManager[tuple[Any, ...]]":
        return self._handle_context("handle_root", None, *args, **kwargs)

    def handle_dependencies(
        self, root_context: "tuple[Any, ...]", *args, **kwargs
    ) -> "contextlib.AbstractContextManager[tuple[Any, ...]]":
        return self._handle_context(
            "handle_dependencies", root_context, *args, **kwargs
        )

    def handle_dependency_set(
        self, dependencies_context: "tuple[Any, ...]", *args, **kwargs
    ) -> "contextlib.AbstractContextManager[tuple[Any, ...]]":
        return self._handle_context(
            "handle_dependency_set", dependencies_context, *args, **kwargs
        )

    def handle_common(
        self, dependency_set_context: "tuple[Any, ...]", *args, **kwargs
    ) -> "contextlib.AbstractContextManager[tuple[Any, ...]]":
        return self._handle_context(
            "handle_common", dependency_set_context, *args, **kwargs
        )

    def handle_common_item(
        self, common_context: "tuple[Any, ...]", *args, **kwargs
    ) -> "contextlib.AbstractContextManager[tuple[Any, ...]]":
        return self._handle_context(
            "handle_common_item", common_context, *args, **kwargs
        )

    def handle_output_types(
        self,
        common_item_or_specific_item_context: "tuple[Any, ...]",
        *args,
        **kwargs,
    ) -> "contextlib.AbstractContextManager[tuple[Any, ...]]":
        return self._handle_context(
            "handle_output_types",
            common_item_or_specific_item_context,
            *args,
            **kwargs,
        )

    def handle_output_type(
        self, output_types_context: "tuple[Any, ...]", *args, **kwargs
    ) -> None:
        return self._handle_no_context(
            "handle_output_type", output_types_context, *args, **kwargs
        )

    def handle_specific(
        self, dependency_set_context: "tuple[Any, ...]", *args, **kwargs
    ) -> "contextlib.AbstractContextManager[tuple[Any, ...]]":
        return self._handle_context(
            "handle_specific", dependency_set_context, *args, **kwargs
        )

    def handle_specific_item(
        self, specific_context: "tuple[Any, ...]", *args, **kwargs
    ) -> "contextlib.AbstractContextManager[tuple[Any, ...]]":
        return self._handle_context(
            "handle_specific_item", specific_context, *args, **kwargs
        )

    def handle_matrices(
        self, specific_item_context: "tuple[Any, ...]", *args, **kwargs
    ) -> "contextlib.AbstractContextManager[tuple[Any, ...]]":
        return self._handle_context(
            "handle_matrices", specific_item_context, *args, **kwargs
        )

    def handle_matrices_item(
        self, matrices_context: "tuple[Any, ...]", *args, **kwargs
    ) -> "contextlib.AbstractContextManager[tuple[Any, ...]]":
        return self._handle_context(
            "handle_matrices_item", matrices_context, *args, **kwargs
        )

    def handle_matrix(
        self, matrices_item_context: "tuple[Any, ...]", *args, **kwargs
    ) -> "contextlib.AbstractContextManager[tuple[Any, ...]]":
        return self._handle_context(
            "handle_matrix", matrices_item_context, *args, **kwargs
        )

    def handle_matrix_item(
        self, matrix_context: "tuple[Any, ...]", *args, **kwargs
    ) -> None:
        return self._handle_no_context(
            "handle_matrix_item", matrix_context, *args, **kwargs
        )

    def handle_packages(
        self,
        common_or_matrices_item_context: "tuple[Any, ...]",
        *args,
        **kwargs,
    ) -> "contextlib.AbstractContextManager[tuple[Any, ...]]":
        return self._handle_context(
            "handle_packages", common_or_matrices_item_context, *args, **kwargs
        )

    def handle_package(
        self, packages_context: "tuple[Any, ...]", *args, **kwargs
    ) -> None:
        return self._handle_no_context(
            "handle_package", packages_context, *args, **kwargs
        )


def traverse_package(
    handler: Handler,
    packages_context: "Any",
    anchors: dict[str, "yaml.Node"],
    used_anchors: set[str],
    node: "yaml.Node",
) -> None:
    if node_has_type(node, "str"):
        anchor = check_and_mark_anchor(anchors, used_anchors, node)
        handler.handle_package(packages_context, anchor, node)


def traverse_packages(
    handler: Handler,
    common_or_matrices_item_context: "Any",
    anchors: dict[str, "yaml.Node"],
    used_anchors: set[str],
    key_node: "yaml.Node",
    node: "yaml.Node",
) -> None:
    if node_has_type(node, "seq"):
        anchor = check_and_mark_anchor(anchors, used_anchors, node)
        with handler.handle_packages(
            common_or_matrices_item_context, anchor, key_node, node
        ) as packages_context:
            for package in node.value:
                traverse_package(
                    handler,
                    packages_context,
                    anchors,
                    used_anchors,
                    package,
                )


def traverse_output_type(
    handler: Handler,
    output_types_context: "Any",
    node: "yaml.Node",
) -> None:
    if node_has_type(node, "str"):
        handler.handle_output_type(output_types_context, node)


def traverse_output_types(
    handler: Handler,
    common_item_or_specific_item_context: "Any",
    key_node: "yaml.Node",
    node: "yaml.Node",
) -> None:
    if node_has_type(node, "seq"):
        with handler.handle_output_types(
            common_item_or_specific_item_context, key_node, node
        ) as output_types_context:
            for item in node.value:
                traverse_output_type(handler, output_types_context, item)
    elif node_has_type(node, "str"):
        with handler.handle_output_types(
            common_item_or_specific_item_context, key_node, node
        ) as output_types_context:
            traverse_output_type(handler, output_types_context, node)


def traverse_common_item(
    handler: Handler,
    common_context: "Any",
    anchors: dict[str, "yaml.Node"],
    used_anchors: set[str],
    node: "yaml.Node",
) -> None:
    if node_has_type(node, "map"):
        with handler.handle_common_item(
            common_context, node
        ) as common_item_context:
            for (
                common_item_key,
                common_item_value,
            ) in node.value:
                if (
                    node_has_type(common_item_key, "str")
                    and common_item_key.value == "output_types"
                ):
                    traverse_output_types(
                        handler,
                        common_item_context,
                        common_item_key,
                        common_item_value,
                    )
                elif (
                    node_has_type(common_item_key, "str")
                    and common_item_key.value == "packages"
                ):
                    traverse_packages(
                        handler,
                        common_item_context,
                        anchors,
                        used_anchors,
                        common_item_key,
                        common_item_value,
                    )


def traverse_common(
    handler: Handler,
    dependency_set_context: "Any",
    anchors: dict[str, "yaml.Node"],
    used_anchors: set[str],
    key_node: "yaml.Node",
    node: "yaml.Node",
) -> None:
    if node_has_type(node, "seq"):
        with handler.handle_common(
            dependency_set_context, key_node, node
        ) as common_context:
            for common_item in node.value:
                traverse_common_item(
                    handler, common_context, anchors, used_anchors, common_item
                )


def traverse_matrix_item(
    handler: Handler,
    matrix_context: "Any",
    key_node: "yaml.Node",
    node: "yaml.Node",
) -> None:
    if node_has_type(node, "str"):
        handler.handle_matrix_item(matrix_context, key_node, node)


def traverse_matrix(
    handler: Handler,
    matrices_item_context: "Any",
    key_node: "yaml.Node",
    node: "yaml.Node",
) -> None:
    if node_has_type(node, "map"):
        with handler.handle_matrix(
            matrices_item_context, key_node, node
        ) as matrices_context:
            for matrix_item_key, matrix_item in node.value:
                traverse_matrix_item(
                    handler,
                    matrices_context,
                    matrix_item_key,
                    matrix_item,
                )
    elif node_has_type(node, "null"):
        with handler.handle_matrix(
            matrices_item_context, key_node, node
        ) as matrices_context:
            pass


def traverse_matrices_item(
    handler: Handler,
    matrices_context: "Any",
    anchors: dict[str, "yaml.Node"],
    used_anchors: set[str],
    node: "yaml.Node",
) -> None:
    if node_has_type(node, "map"):
        with handler.handle_matrices_item(
            matrices_context, node
        ) as matrices_item_context:
            for matrix_key, matrix_value in node.value:
                if (
                    node_has_type(matrix_key, "str")
                    and matrix_key.value == "packages"
                ):
                    traverse_packages(
                        handler,
                        matrices_item_context,
                        anchors,
                        used_anchors,
                        matrix_key,
                        matrix_value,
                    )
                elif (
                    node_has_type(matrix_key, "str")
                    and matrix_key.value == "matrix"
                ):
                    traverse_matrix(
                        handler,
                        matrices_item_context,
                        matrix_key,
                        matrix_value,
                    )


def traverse_matrices(
    handler: Handler,
    specific_item_context: "Any",
    anchors: dict[str, "yaml.Node"],
    used_anchors: set[str],
    key_node: "yaml.Node",
    node: "yaml.Node",
) -> None:
    if node_has_type(node, "seq"):
        with handler.handle_matrices(
            specific_item_context, key_node, node
        ) as matrices_context:
            for item in node.value:
                traverse_matrices_item(
                    handler, matrices_context, anchors, used_anchors, item
                )


def traverse_specific_item(
    handler: Handler,
    specific_context: "Any",
    anchors: dict[str, "yaml.Node"],
    used_anchors: set[str],
    node: "yaml.Node",
) -> None:
    if node_has_type(node, "map"):
        with handler.handle_specific_item(
            specific_context, node
        ) as specific_item_context:
            for (
                specific_item_key,
                specific_item_value,
            ) in node.value:
                if (
                    node_has_type(specific_item_key, "str")
                    and specific_item_key.value == "output_types"
                ):
                    traverse_output_types(
                        handler,
                        specific_item_context,
                        specific_item_key,
                        specific_item_value,
                    )
                elif (
                    node_has_type(specific_item_key, "str")
                    and specific_item_key.value == "matrices"
                ):
                    traverse_matrices(
                        handler,
                        specific_item_context,
                        anchors,
                        used_anchors,
                        specific_item_key,
                        specific_item_value,
                    )


def traverse_specific(
    handler: Handler,
    dependency_set_context: "Any",
    anchors: dict[str, "yaml.Node"],
    used_anchors: set[str],
    key_node: "yaml.Node",
    node: "yaml.Node",
) -> None:
    if node_has_type(node, "seq"):
        with handler.handle_specific(
            dependency_set_context, key_node, node
        ) as specific_context:
            for specific_item in node.value:
                traverse_specific_item(
                    handler,
                    specific_context,
                    anchors,
                    used_anchors,
                    specific_item,
                )


def traverse_dependency_set(
    handler: Handler,
    dependencies_context: "Any",
    anchors: dict[str, "yaml.Node"],
    used_anchors: set[str],
    key_node: "yaml.Node",
    node: "yaml.Node",
) -> None:
    if node_has_type(node, "map"):
        with handler.handle_dependency_set(
            dependencies_context, key_node, node
        ) as dependency_set_context:
            for (
                dependency_key,
                dependency_value,
            ) in node.value:
                if node_has_type(dependency_key, "str"):
                    if dependency_key.value == "common":
                        traverse_common(
                            handler,
                            dependency_set_context,
                            anchors,
                            used_anchors,
                            dependency_key,
                            dependency_value,
                        )
                    elif dependency_key.value == "specific":
                        traverse_specific(
                            handler,
                            dependency_set_context,
                            anchors,
                            used_anchors,
                            dependency_key,
                            dependency_value,
                        )


def traverse_dependencies(
    handler: Handler,
    root_context: "Any",
    anchors: dict[str, "yaml.Node"],
    used_anchors: set[str],
    key_node: "yaml.Node",
    node: "yaml.Node",
) -> None:
    if node_has_type(node, "map"):
        with handler.handle_dependencies(
            root_context, key_node, node
        ) as dependencies_context:
            for dependencies_key, dependencies_value in node.value:
                traverse_dependency_set(
                    handler,
                    dependencies_context,
                    anchors,
                    used_anchors,
                    dependencies_key,
                    dependencies_value,
                )


def traverse_root(
    handler: Handler,
    anchors: dict[str, "yaml.Node"],
    used_anchors: set[str],
    node: "yaml.Node",
) -> None:
    if node_has_type(node, "map"):
        with handler.handle_root(node) as root_context:
            for root_key, root_value in node.value:
                if (
                    node_has_type(root_key, "str")
                    and root_key.value == "dependencies"
                ):
                    traverse_dependencies(
                        handler,
                        root_context,
                        anchors,
                        used_anchors,
                        root_key,
                        root_value,
                    )


def traverse_dependencies_yaml(handler: Handler, content: str) -> None:
    loader = AnchorPreservingLoader(content)
    try:
        root = loader.get_single_node()
        assert root is not None
    finally:
        loader.dispose()
    traverse_root(handler, loader.document_anchors[0], set(), root)


def is_python_output_type(output_type: str) -> bool:
    return output_type in {"requirements", "constraints", "pyproject"}
