# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock, Mock, call, patch

import pytest
import yaml

from rapids_pre_commit_hooks.utils.yaml import Anchor, AnchorType
from rapids_pre_commit_hooks.utils import dependencies_yaml
from rapids_pre_commit_hooks_test_utils import (
    find_yaml_node_for_span,
    parse_named_spans,
)


class TestChainedHandler:
    @pytest.mark.parametrize(
        ["hook_name", "use_context", "hook_args"],
        [
            pytest.param(
                "handle_root",
                False,
                (Mock(),),
                id="handle_root",
            ),
            pytest.param(
                "handle_dependencies",
                True,
                (Mock(), Mock()),
                id="handle_dependencies",
            ),
            pytest.param(
                "handle_dependency_set",
                True,
                (Mock(), Mock()),
                id="handle_dependency_set",
            ),
            pytest.param(
                "handle_common",
                True,
                (Mock(), Mock()),
                id="handle_common",
            ),
            pytest.param(
                "handle_common_item",
                True,
                (Mock(),),
                id="handle_common_item",
            ),
            pytest.param(
                "handle_output_types",
                True,
                (Mock(), Mock()),
                id="handle_output_types",
            ),
            pytest.param(
                "handle_specific",
                True,
                (Mock(), Mock()),
                id="handle_specific",
            ),
            pytest.param(
                "handle_specific_item",
                True,
                (Mock(),),
                id="handle_specific_item",
            ),
            pytest.param(
                "handle_matrices",
                True,
                (Mock(), Mock()),
                id="handle_matrices",
            ),
            pytest.param(
                "handle_matrices_item",
                True,
                (Mock(),),
                id="handle_matrices_item",
            ),
            pytest.param(
                "handle_matrix",
                True,
                (Mock(), Mock()),
                id="handle_matrix",
            ),
            pytest.param(
                "handle_packages",
                True,
                (Mock(), Mock(), Mock()),
                id="handle_packages",
            ),
        ],
    )
    def test_context(self, hook_name, use_context, hook_args):
        manager = MagicMock()

        chained_handler = dependencies_yaml.ChainedHandler()
        chained_handler.add_handler(manager.handler_1)
        chained_handler.add_handler(manager.handler_2)

        context_arg, context_arg_1, context_arg_2 = (
            (
                ((manager.context_1, manager.context_2),),
                (manager.context_1,),
                (manager.context_2,),
            )
            if use_context
            else ((), (), ())
        )

        expected_context = (
            getattr(manager.handler_1, hook_name)().__enter__(),
            getattr(manager.handler_2, hook_name)().__enter__(),
        )
        expected_calls = [
            getattr(call.handler_1, hook_name)(*context_arg_1, *hook_args),
            getattr(call.handler_1, hook_name)().__enter__(
                getattr(manager.handler_1, hook_name)()
            ),
            getattr(call.handler_2, hook_name)(*context_arg_2, *hook_args),
            getattr(call.handler_2, hook_name)().__enter__(
                getattr(manager.handler_2, hook_name)()
            ),
            getattr(call.handler_2, hook_name)().__exit__(
                getattr(manager.handler_2, hook_name)(), None, None, None
            ),
            getattr(call.handler_1, hook_name)().__exit__(
                getattr(manager.handler_1, hook_name)(), None, None, None
            ),
        ]
        manager.reset_mock()

        with getattr(chained_handler, hook_name)(
            *context_arg, *hook_args
        ) as context:
            assert context == expected_context

        assert manager.mock_calls == expected_calls

    @pytest.mark.parametrize(
        ["hook_name", "hook_args"],
        [
            pytest.param(
                "handle_matrix_item",
                (Mock(), Mock()),
                id="handle_matrix_item",
            ),
            pytest.param(
                "handle_package",
                (Mock(), Mock()),
                id="handle_package",
            ),
            pytest.param(
                "handle_output_type",
                (Mock(),),
                id="handle_output_type",
            ),
        ],
    )
    def test_no_context(self, hook_name, hook_args):
        manager = MagicMock()

        chained_handler = dependencies_yaml.ChainedHandler()
        chained_handler.add_handler(manager.handler_1)
        chained_handler.add_handler(manager.handler_2)

        expected_calls = [
            getattr(call.handler_1, hook_name)(manager.context_1, *hook_args),
            getattr(call.handler_2, hook_name)(manager.context_2, *hook_args),
        ]
        manager.reset_mock()

        getattr(chained_handler, hook_name)(
            (manager.context_1, manager.context_2), *hook_args
        )

        assert manager.mock_calls == expected_calls


@pytest.mark.parametrize(
    ["content", "used_anchors", "anchor"],
    [
        pytest.param(
            """\
            + - lib1
            :   ~~~~node
            """,
            set(),
            None,
            id="no-anchor",
        ),
        pytest.param(
            """\
            + - &lib1 lib1
            :   ~~~~~~~~~~node
            :   ~~~~~~~~~~anchors.lib1
            + - *lib1
            """,
            set(),
            Anchor(AnchorType.DEFINITION, "lib1"),
            id="anchor-definition",
        ),
        pytest.param(
            """\
            + - &lib1 lib1
            :   ~~~~~~~~~~node
            :   ~~~~~~~~~~anchors.lib1
            + - *lib1
            """,
            {"lib1"},
            Anchor(AnchorType.REFERENCE, "lib1"),
            id="anchor-reference",
        ),
    ],
)
def test_traverse_package(content, used_anchors, anchor):
    content, spans = parse_named_spans(content)
    composed = yaml.SafeLoader(content).get_single_node()
    package = find_yaml_node_for_span(composed, spans["node"])
    packages_context = Mock()
    manager = MagicMock()

    expected_calls = [
        call.handler.handle_package(packages_context, anchor, package),
    ]
    manager.reset_mock()

    anchors = {
        name: find_yaml_node_for_span(composed, span)
        for name, span in spans.get("anchors", {}).items()
    }
    dependencies_yaml.traverse_package(
        manager.handler, packages_context, anchors, used_anchors, package
    )

    assert manager.mock_calls == expected_calls


@pytest.mark.parametrize(
    ["content", "used_anchors", "used_anchors_after", "anchor"],
    [
        pytest.param(
            """\
            + packages:
            : ~~~~~~~~packages_key
            +     - lib1
            :     >packages
            +     - lib2
            :            !packages
            """,
            set(),
            set(),
            None,
            id="no-anchor",
        ),
        pytest.param(
            """\
            + - packages: &packages
            :   ~~~~~~~~packages_key
            :             >packages
            :             >anchors.packages
            +     - lib1
            +     - lib2
            :            !packages
            :            !anchors.packages
            + - packages: *packages
            """,
            set(),
            {"packages"},
            Anchor(AnchorType.DEFINITION, "packages"),
            id="anchor-definition",
        ),
        pytest.param(
            """\
            + - packages: &packages
            :             >packages
            :             >anchors.packages
            +     - lib1
            +     - lib2
            :            !packages
            :            !anchors.packages
            + - packages: *packages
            :   ~~~~~~~~packages_key
            """,
            {"packages"},
            {"packages"},
            Anchor(AnchorType.REFERENCE, "packages"),
            id="anchor-reference",
        ),
    ],
)
def test_traverse_packages(content, used_anchors, used_anchors_after, anchor):
    content, spans = parse_named_spans(content)
    composed = yaml.SafeLoader(content).get_single_node()
    packages_key = find_yaml_node_for_span(composed, spans["packages_key"])
    packages = find_yaml_node_for_span(composed, spans["packages"])
    item_context = Mock()
    manager = MagicMock()

    anchors = {
        name: find_yaml_node_for_span(composed, span)
        for name, span in spans.get("anchors", {}).items()
    }
    expected_calls = [
        call.handler.handle_packages(
            item_context, anchor, packages_key, packages
        ),
        call.handler.handle_packages().__enter__(),
        call.traverse_package(
            manager.handler,
            manager.handler.handle_packages().__enter__(),
            anchors,
            used_anchors_after,
            packages.value[0],
        ),
        call.traverse_package(
            manager.handler,
            manager.handler.handle_packages().__enter__(),
            anchors,
            used_anchors_after,
            packages.value[1],
        ),
        call.handler.handle_packages().__exit__(None, None, None),
    ]
    manager.reset_mock()

    with (
        patch(
            "rapids_pre_commit_hooks.utils.dependencies_yaml.traverse_package",
            manager.traverse_package,
        ),
    ):
        dependencies_yaml.traverse_packages(
            manager.handler,
            item_context,
            anchors,
            used_anchors,
            packages_key,
            packages,
        )

    assert manager.mock_calls == expected_calls


def test_traverse_output_type():
    output_types = yaml.SafeLoader("""\
    [requirements]
    """).get_single_node()
    output_type = output_types.value[0]
    output_types_context = Mock()
    manager = MagicMock()

    expected_calls = [
        call.handler.handle_output_type(output_types_context, output_type),
    ]
    manager.reset_mock()

    dependencies_yaml.traverse_output_type(
        manager.handler, output_types_context, output_type
    )

    assert manager.mock_calls == expected_calls


@pytest.mark.parametrize(
    ["content"],
    [
        pytest.param(
            """\
            + output_types: pyproject
            : ~~~~~~~~~~~~key_node
            :               ~~~~~~~~~node
            :               ~~~~~~~~~items.0
            """,
            id="string-item",
        ),
        pytest.param(
            """\
            + output_types: [requirements, pyproject]
            : ~~~~~~~~~~~~key_node
            :               ~~~~~~~~~~~~~~~~~~~~~~~~~node
            :                ~~~~~~~~~~~~items.0
            :                              ~~~~~~~~~items.1
            """,
            id="list",
        ),
        pytest.param(
            """\
            + output_types: []
            : ~~~~~~~~~~~~key_node
            :               ~~node
            """,
            id="empty-list",
        ),
    ],
)
def test_traverse_output_types(content):
    content, spans = parse_named_spans(content)
    item = yaml.SafeLoader(content).get_single_node()
    output_types_key = find_yaml_node_for_span(item, spans["key_node"])
    output_types = find_yaml_node_for_span(item, spans["node"])
    item_context = Mock()
    manager = MagicMock()

    expected_calls = [
        call.handler.handle_output_types(
            item_context, output_types_key, output_types
        ),
        call.handler.handle_output_types().__enter__(),
        *(
            call.traverse_output_type(
                manager.handler,
                manager.handler.handle_output_types().__enter__(),
                find_yaml_node_for_span(item, output_type_span),
            )
            for output_type_span in spans.get("items", [])
        ),
        call.handler.handle_output_types().__exit__(None, None, None),
    ]
    manager.reset_mock()

    with (
        patch(
            "rapids_pre_commit_hooks.utils.dependencies_yaml.traverse_output_type",
            manager.traverse_output_type,
        ),
    ):
        dependencies_yaml.traverse_output_types(
            manager.handler, item_context, output_types_key, output_types
        )

    assert manager.mock_calls == expected_calls


def test_traverse_common_item():
    common = yaml.SafeLoader("""\
    - output_types: pyproject
      packages: []
    """).get_single_node()
    common_item = common.value[0]
    common_context = Mock()
    manager = MagicMock()

    expected_calls = [
        call.handler.handle_common_item(common_context, common_item),
        call.handler.handle_common_item().__enter__(),
        call.traverse_output_types(
            manager.handler,
            manager.handler.handle_common_item().__enter__(),
            common_item.value[0][0],
            common_item.value[0][1],
        ),
        call.traverse_packages(
            manager.handler,
            manager.handler.handle_common_item().__enter__(),
            {},
            set(),
            common_item.value[1][0],
            common_item.value[1][1],
        ),
        call.handler.handle_common_item().__exit__(None, None, None),
    ]
    manager.reset_mock()

    with (
        patch(
            "rapids_pre_commit_hooks.utils.dependencies_yaml.traverse_output_types",
            manager.traverse_output_types,
        ),
        patch(
            "rapids_pre_commit_hooks.utils.dependencies_yaml.traverse_packages",
            manager.traverse_packages,
        ),
    ):
        dependencies_yaml.traverse_common_item(
            manager.handler, common_context, {}, set(), common_item
        )

    assert manager.mock_calls == expected_calls


def test_traverse_common():
    dependency_set = yaml.SafeLoader("""\
    common:
        - {}
        - {}
    """).get_single_node()
    common_key, common = dependency_set.value[0]
    dependency_set_context = Mock()
    manager = MagicMock()

    expected_calls = [
        call.handler.handle_common(dependency_set_context, common_key, common),
        call.handler.handle_common().__enter__(),
        call.traverse_common_item(
            manager.handler,
            manager.handler.handle_common().__enter__(),
            {},
            set(),
            common.value[0],
        ),
        call.traverse_common_item(
            manager.handler,
            manager.handler.handle_common().__enter__(),
            {},
            set(),
            common.value[1],
        ),
        call.handler.handle_common().__exit__(None, None, None),
    ]
    manager.reset_mock()

    with (
        patch(
            "rapids_pre_commit_hooks.utils.dependencies_yaml.traverse_common_item",
            manager.traverse_common_item,
        ),
    ):
        dependencies_yaml.traverse_common(
            manager.handler,
            dependency_set_context,
            {},
            set(),
            common_key,
            common,
        )

    assert manager.mock_calls == expected_calls


def test_traverse_matrix_item():
    matrix = yaml.SafeLoader("""\
    value_1: "true"
    """).get_single_node()
    matrix_item_key, matrix_item = matrix.value[0]
    matrix_context = Mock()
    manager = MagicMock()

    expected_calls = [
        call.handler.handle_matrix_item(
            matrix_context, matrix_item_key, matrix_item
        ),
    ]
    manager.reset_mock()

    dependencies_yaml.traverse_matrix_item(
        manager.handler,
        matrix_context,
        matrix_item_key,
        matrix_item,
    )

    assert manager.mock_calls == expected_calls


def test_traverse_matrix():
    matrices_item = yaml.SafeLoader("""\
    matrix:
        value_1: "true"
        value_2: "true"
    """).get_single_node()
    matrix_key, matrix = matrices_item.value[0]
    matrices_item_context = Mock()
    manager = MagicMock()

    expected_calls = [
        call.handler.handle_matrix(matrices_item_context, matrix_key, matrix),
        call.handler.handle_matrix().__enter__(),
        call.traverse_matrix_item(
            manager.handler,
            manager.handler.handle_matrix().__enter__(),
            matrix.value[0][0],
            matrix.value[0][1],
        ),
        call.traverse_matrix_item(
            manager.handler,
            manager.handler.handle_matrix().__enter__(),
            matrix.value[1][0],
            matrix.value[1][1],
        ),
        call.handler.handle_matrix().__exit__(None, None, None),
    ]
    manager.reset_mock()

    with (
        patch(
            "rapids_pre_commit_hooks.utils.dependencies_yaml.traverse_matrix_item",
            manager.traverse_matrix_item,
        ),
    ):
        dependencies_yaml.traverse_matrix(
            manager.handler,
            matrices_item_context,
            matrix_key,
            matrix,
        )

    assert manager.mock_calls == expected_calls


def test_traverse_matrices_item():
    matrices = yaml.SafeLoader("""\
    - matrix: {}
      packages: []
    """).get_single_node()
    matrices_item = matrices.value[0]
    matrices_context = Mock()
    manager = MagicMock()

    expected_calls = [
        call.handler.handle_matrices_item(matrices_context, matrices_item),
        call.handler.handle_matrices_item().__enter__(),
        call.traverse_matrix(
            manager.handler,
            manager.handler.handle_matrices_item().__enter__(),
            matrices_item.value[0][0],
            matrices_item.value[0][1],
        ),
        call.traverse_packages(
            manager.handler,
            manager.handler.handle_matrices_item().__enter__(),
            {},
            set(),
            matrices_item.value[1][0],
            matrices_item.value[1][1],
        ),
        call.handler.handle_matrices_item().__exit__(None, None, None),
    ]
    manager.reset_mock()

    with (
        patch(
            "rapids_pre_commit_hooks.utils.dependencies_yaml.traverse_matrix",
            manager.traverse_matrix,
        ),
        patch(
            "rapids_pre_commit_hooks.utils.dependencies_yaml.traverse_packages",
            manager.traverse_packages,
        ),
    ):
        dependencies_yaml.traverse_matrices_item(
            manager.handler, matrices_context, {}, set(), matrices_item
        )

    assert manager.mock_calls == expected_calls


def test_traverse_matrices():
    specific_item = yaml.SafeLoader("""\
    matrices:
        - {}
        - {}
        - {}
    """).get_single_node()
    matrices_key, matrices = specific_item.value[0]
    specific_item_context = Mock()
    manager = MagicMock()

    expected_calls = [
        call.handler.handle_matrices(
            specific_item_context, matrices_key, matrices
        ),
        call.handler.handle_matrices().__enter__(),
        call.traverse_matrices_item(
            manager.handler,
            manager.handler.handle_matrices().__enter__(),
            {},
            set(),
            matrices.value[0],
        ),
        call.traverse_matrices_item(
            manager.handler,
            manager.handler.handle_matrices().__enter__(),
            {},
            set(),
            matrices.value[1],
        ),
        call.traverse_matrices_item(
            manager.handler,
            manager.handler.handle_matrices().__enter__(),
            {},
            set(),
            matrices.value[2],
        ),
        call.handler.handle_matrices().__exit__(None, None, None),
    ]
    manager.reset_mock()

    with (
        patch(
            "rapids_pre_commit_hooks.utils.dependencies_yaml.traverse_matrices_item",
            manager.traverse_matrices_item,
        ),
    ):
        dependencies_yaml.traverse_matrices(
            manager.handler,
            specific_item_context,
            {},
            set(),
            matrices_key,
            matrices,
        )

    assert manager.mock_calls == expected_calls


def test_traverse_specific_item():
    specific = yaml.SafeLoader("""\
    - output_types: pyproject
      matrices: []
    """).get_single_node()
    specific_item = specific.value[0]
    specific_context = Mock()
    manager = MagicMock()

    expected_calls = [
        call.handler.handle_specific_item(specific_context, specific_item),
        call.handler.handle_specific_item().__enter__(),
        call.traverse_output_types(
            manager.handler,
            manager.handler.handle_specific_item().__enter__(),
            specific_item.value[0][0],
            specific_item.value[0][1],
        ),
        call.traverse_matrices(
            manager.handler,
            manager.handler.handle_specific_item().__enter__(),
            {},
            set(),
            specific_item.value[1][0],
            specific_item.value[1][1],
        ),
        call.handler.handle_specific_item().__exit__(None, None, None),
    ]
    manager.reset_mock()

    with (
        patch(
            "rapids_pre_commit_hooks.utils.dependencies_yaml.traverse_output_types",
            manager.traverse_output_types,
        ),
        patch(
            "rapids_pre_commit_hooks.utils.dependencies_yaml.traverse_matrices",
            manager.traverse_matrices,
        ),
    ):
        dependencies_yaml.traverse_specific_item(
            manager.handler, specific_context, {}, set(), specific_item
        )

    assert manager.mock_calls == expected_calls


def test_traverse_specific():
    dependency_set = yaml.SafeLoader("""\
    specific:
        - {}
        - {}
        - {}
    """).get_single_node()
    specific_key, specific = dependency_set.value[0]
    dependency_set_context = Mock()
    manager = MagicMock()

    expected_calls = [
        call.handler.handle_specific(
            dependency_set_context, specific_key, specific
        ),
        call.handler.handle_specific().__enter__(),
        call.traverse_specific_item(
            manager.handler,
            manager.handler.handle_specific().__enter__(),
            {},
            set(),
            specific.value[0],
        ),
        call.traverse_specific_item(
            manager.handler,
            manager.handler.handle_specific().__enter__(),
            {},
            set(),
            specific.value[1],
        ),
        call.traverse_specific_item(
            manager.handler,
            manager.handler.handle_specific().__enter__(),
            {},
            set(),
            specific.value[2],
        ),
        call.handler.handle_specific().__exit__(None, None, None),
    ]
    manager.reset_mock()

    with (
        patch(
            "rapids_pre_commit_hooks.utils.dependencies_yaml.traverse_specific_item",
            manager.traverse_specific_item,
        ),
    ):
        dependencies_yaml.traverse_specific(
            manager.handler,
            dependency_set_context,
            {},
            set(),
            specific_key,
            specific,
        )

    assert manager.mock_calls == expected_calls


def test_traverse_dependency_set():
    dependencies = yaml.SafeLoader("""\
    dependency_set_1:
        common: {}
        specific: {}
    """).get_single_node()
    dependency_set_key, dependency_set = dependencies.value[0]
    dependencies_context = Mock()
    manager = MagicMock()

    expected_calls = [
        call.handler.handle_dependency_set(
            dependencies_context, dependency_set_key, dependency_set
        ),
        call.handler.handle_dependency_set().__enter__(),
        call.traverse_common(
            manager.handler,
            manager.handler.handle_dependency_set().__enter__(),
            {},
            set(),
            dependency_set.value[0][0],
            dependency_set.value[0][1],
        ),
        call.traverse_specific(
            manager.handler,
            manager.handler.handle_dependency_set().__enter__(),
            {},
            set(),
            dependency_set.value[1][0],
            dependency_set.value[1][1],
        ),
        call.handler.handle_dependency_set().__exit__(None, None, None),
    ]
    manager.reset_mock()

    with (
        patch(
            "rapids_pre_commit_hooks.utils.dependencies_yaml.traverse_common",
            manager.traverse_common,
        ),
        patch(
            "rapids_pre_commit_hooks.utils.dependencies_yaml.traverse_specific",
            manager.traverse_specific,
        ),
    ):
        dependencies_yaml.traverse_dependency_set(
            manager.handler,
            dependencies_context,
            {},
            set(),
            dependency_set_key,
            dependency_set,
        )

    assert manager.mock_calls == expected_calls


def test_traverse_dependencies():
    root = yaml.SafeLoader("""\
    dependencies:
        dependency_set_1: {}
        dependency_set_2: {}
    """).get_single_node()
    dependencies_key, dependencies = root.value[0]
    root_context = Mock()
    manager = MagicMock()

    expected_calls = [
        call.handler.handle_dependencies(
            root_context, dependencies_key, dependencies
        ),
        call.handler.handle_dependencies().__enter__(),
        call.traverse_dependency_set(
            manager.handler,
            manager.handler.handle_dependencies().__enter__(),
            {},
            set(),
            dependencies.value[0][0],
            dependencies.value[0][1],
        ),
        call.traverse_dependency_set(
            manager.handler,
            manager.handler.handle_dependencies().__enter__(),
            {},
            set(),
            dependencies.value[1][0],
            dependencies.value[1][1],
        ),
        call.handler.handle_dependencies().__exit__(None, None, None),
    ]
    manager.reset_mock()

    with patch(
        "rapids_pre_commit_hooks.utils.dependencies_yaml.traverse_dependency_set",
        manager.traverse_dependency_set,
    ):
        dependencies_yaml.traverse_dependencies(
            manager.handler,
            root_context,
            {},
            set(),
            dependencies_key,
            dependencies,
        )

    assert manager.mock_calls == expected_calls


def test_traverse_root():
    root = yaml.SafeLoader("""\
    files: {}
    channels: []
    dependencies: {}
    """).get_single_node()
    manager = MagicMock()

    expected_calls = [
        call.handler.handle_root(root),
        call.handler.handle_root().__enter__(),
        call.traverse_dependencies(
            manager.handler,
            manager.handler.handle_root().__enter__(),
            {},
            set(),
            root.value[2][0],
            root.value[2][1],
        ),
        call.handler.handle_root().__exit__(None, None, None),
    ]
    manager.reset_mock()

    with patch(
        "rapids_pre_commit_hooks.utils.dependencies_yaml.traverse_dependencies",
        manager.traverse_dependencies,
    ):
        dependencies_yaml.traverse_root(manager.handler, {}, set(), root)

    assert manager.mock_calls == expected_calls


@pytest.mark.parametrize(
    ["output_type", "is_python"],
    [
        pytest.param(
            output_type,
            is_python,
            id=output_type,
        )
        for output_type, is_python in [
            ("conda", False),
            ("requirements", True),
            ("constraints", True),
            ("pyproject", True),
        ]
    ],
)
def test_is_python_output_type(output_type, is_python):
    assert dependencies_yaml.is_python_output_type(output_type) is is_python
