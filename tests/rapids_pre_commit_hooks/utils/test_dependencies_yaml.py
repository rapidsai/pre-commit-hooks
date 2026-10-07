# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock, Mock, call, patch

import pytest

from rapids_pre_commit_hooks.utils import dependencies_yaml
from rapids_pre_commit_hooks.utils.yaml import (
    Anchor,
    AnchorType,
    load_with_anchors,
)
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
                "handle_files",
                True,
                (Mock(), Mock()),
                id="handle_files",
            ),
            pytest.param(
                "handle_files_item",
                True,
                (Mock(), Mock()),
                id="handle_files_item",
            ),
            pytest.param(
                "handle_file_output",
                True,
                (Mock(), Mock()),
                id="handle_file_output",
            ),
            pytest.param(
                "handle_extras",
                True,
                (Mock(), Mock()),
                id="handle_extras",
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
            pytest.param(
                "handle_file_output_item",
                (Mock(),),
                id="handle_file_output_item",
            ),
            pytest.param(
                "handle_extras_table",
                (Mock(), Mock()),
                id="handle_extras_table",
            ),
            pytest.param(
                "handle_extras_key",
                (Mock(), Mock()),
                id="handle_extras_key",
            ),
            pytest.param(
                "handle_pyproject_dir",
                (Mock(), Mock()),
                id="handle_pyproject_dir",
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


def test_traverse_file_output_item():
    content, spans = parse_named_spans(
        """\
        + [pyproject]
        :  ~~~~~~~~~file_output_item
        """
    )
    file_output, _ = load_with_anchors(content)
    file_output_item = find_yaml_node_for_span(
        file_output, spans["file_output_item"]
    )
    file_output_context = Mock()
    manager = MagicMock()

    expected_calls = [
        call.handler.handle_file_output_item(
            file_output_context, file_output_item
        ),
    ]
    manager.reset_mock()

    dependencies_yaml.traverse_file_output_item(
        manager.handler, file_output_context, file_output_item
    )

    assert manager.mock_calls == expected_calls


@pytest.mark.parametrize(
    ["content"],
    [
        pytest.param(
            """\
            + output: pyproject
            : ~~~~~~key_node
            :         ~~~~~~~~~node
            :         ~~~~~~~~~items.0
            """,
            id="string-item",
        ),
        pytest.param(
            """\
            + output: [requirements, pyproject]
            : ~~~~~~key_node
            :         ~~~~~~~~~~~~~~~~~~~~~~~~~node
            :          ~~~~~~~~~~~~items.0
            :                        ~~~~~~~~~items.1
            """,
            id="list",
        ),
        pytest.param(
            """\
            + output: []
            : ~~~~~~key_node
            :         ~~node
            """,
            id="empty-list",
        ),
    ],
)
def test_traverse_file_output(content):
    content, spans = parse_named_spans(content)
    files_item, _ = load_with_anchors(content)
    output_key = find_yaml_node_for_span(files_item, spans["key_node"])
    output = find_yaml_node_for_span(files_item, spans["node"])
    files_item_context = Mock()
    manager = MagicMock()

    expected_calls = [
        call.handler.handle_file_output(
            files_item_context, output_key, output
        ),
        call.handler.handle_file_output().__enter__(),
        *(
            call.traverse_file_output_item(
                manager.handler,
                manager.handler.handle_file_output().__enter__(),
                find_yaml_node_for_span(files_item, item_span),
            )
            for item_span in spans.get("items", [])
        ),
        call.handler.handle_file_output().__exit__(None, None, None),
    ]
    manager.reset_mock()

    with patch(
        "rapids_pre_commit_hooks.utils.dependencies_yaml."
        "traverse_file_output_item",
        manager.traverse_file_output_item,
    ):
        dependencies_yaml.traverse_file_output(
            manager.handler, files_item_context, output_key, output
        )

    assert manager.mock_calls == expected_calls


@pytest.mark.parametrize(
    ["function_name", "handler_name", "content"],
    [
        pytest.param(
            "traverse_extras_table",
            "handle_extras_table",
            """\
            + table: project.optional-dependencies
            : ~~~~~key
            :        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~value
            """,
            id="extras-table",
        ),
        pytest.param(
            "traverse_extras_key",
            "handle_extras_key",
            """\
            + key: test
            : ~~~key
            :      ~~~~value
            """,
            id="extras-key",
        ),
        pytest.param(
            "traverse_pyproject_dir",
            "handle_pyproject_dir",
            """\
            + pyproject_dir: python
            : ~~~~~~~~~~~~~key
            :                ~~~~~~value
            """,
            id="pyproject-dir",
        ),
    ],
)
def test_traverse_string_value(function_name, handler_name, content):
    content, spans = parse_named_spans(content)
    parent, _ = load_with_anchors(content)
    key = find_yaml_node_for_span(parent, spans["key"])
    value = find_yaml_node_for_span(parent, spans["value"])
    parent_context = Mock()
    manager = MagicMock()

    expected_calls = [
        getattr(call.handler, handler_name)(parent_context, key, value),
    ]
    manager.reset_mock()

    getattr(dependencies_yaml, function_name)(
        manager.handler, parent_context, key, value
    )

    assert manager.mock_calls == expected_calls


def test_traverse_extras():
    content, spans = parse_named_spans(
        """\
        + extras:
        : ~~~~~~extras_key
        +     table: project.optional-dependencies
        :     >extras
        :     ~~~~~table_key
        :            ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~table
        +     key: test
        :               !extras
        :     ~~~key_key
        :          ~~~~key
        """
    )
    files_item, _ = load_with_anchors(content)
    extras_key = find_yaml_node_for_span(files_item, spans["extras_key"])
    extras = find_yaml_node_for_span(files_item, spans["extras"])
    files_item_context = Mock()
    manager = MagicMock()

    expected_calls = [
        call.handler.handle_extras(files_item_context, extras_key, extras),
        call.handler.handle_extras().__enter__(),
        call.traverse_extras_table(
            manager.handler,
            manager.handler.handle_extras().__enter__(),
            find_yaml_node_for_span(files_item, spans["table_key"]),
            find_yaml_node_for_span(files_item, spans["table"]),
        ),
        call.traverse_extras_key(
            manager.handler,
            manager.handler.handle_extras().__enter__(),
            find_yaml_node_for_span(files_item, spans["key_key"]),
            find_yaml_node_for_span(files_item, spans["key"]),
        ),
        call.handler.handle_extras().__exit__(None, None, None),
    ]
    manager.reset_mock()

    with (
        patch(
            "rapids_pre_commit_hooks.utils.dependencies_yaml."
            "traverse_extras_table",
            manager.traverse_extras_table,
        ),
        patch(
            "rapids_pre_commit_hooks.utils.dependencies_yaml."
            "traverse_extras_key",
            manager.traverse_extras_key,
        ),
    ):
        dependencies_yaml.traverse_extras(
            manager.handler, files_item_context, extras_key, extras
        )

    assert manager.mock_calls == expected_calls


def test_traverse_files_item():
    content, spans = parse_named_spans(
        """\
        + test:
        : ~~~~files_item_key
        +     output: pyproject
        :     >files_item
        :     ~~~~~~output_key
        :             ~~~~~~~~~output
        +     extras: {}
        :     ~~~~~~extras_key
        :             ~~extras
        +     pyproject_dir: python
        :     ~~~~~~~~~~~~~pyproject_dir_key
        :                    ~~~~~~pyproject_dir
        +     includes: []
        :                  !files_item
        """
    )
    files, _ = load_with_anchors(content)
    files_item_key = find_yaml_node_for_span(files, spans["files_item_key"])
    files_item = find_yaml_node_for_span(files, spans["files_item"])
    files_context = Mock()
    manager = MagicMock()

    expected_calls = [
        call.handler.handle_files_item(
            files_context, files_item_key, files_item
        ),
        call.handler.handle_files_item().__enter__(),
        call.traverse_file_output(
            manager.handler,
            manager.handler.handle_files_item().__enter__(),
            find_yaml_node_for_span(files, spans["output_key"]),
            find_yaml_node_for_span(files, spans["output"]),
        ),
        call.traverse_extras(
            manager.handler,
            manager.handler.handle_files_item().__enter__(),
            find_yaml_node_for_span(files, spans["extras_key"]),
            find_yaml_node_for_span(files, spans["extras"]),
        ),
        call.traverse_pyproject_dir(
            manager.handler,
            manager.handler.handle_files_item().__enter__(),
            find_yaml_node_for_span(files, spans["pyproject_dir_key"]),
            find_yaml_node_for_span(files, spans["pyproject_dir"]),
        ),
        call.handler.handle_files_item().__exit__(None, None, None),
    ]
    manager.reset_mock()

    with (
        patch(
            "rapids_pre_commit_hooks.utils.dependencies_yaml."
            "traverse_file_output",
            manager.traverse_file_output,
        ),
        patch(
            "rapids_pre_commit_hooks.utils.dependencies_yaml.traverse_extras",
            manager.traverse_extras,
        ),
        patch(
            "rapids_pre_commit_hooks.utils.dependencies_yaml."
            "traverse_pyproject_dir",
            manager.traverse_pyproject_dir,
        ),
    ):
        dependencies_yaml.traverse_files_item(
            manager.handler, files_context, files_item_key, files_item
        )

    assert manager.mock_calls == expected_calls


def test_traverse_files():
    content, spans = parse_named_spans(
        """\
        + files:
        : ~~~~~files_key
        +     test: {}
        :     >files
        :     ~~~~test_key
        :           ~~test
        +     all: {}
        :             !files
        :     ~~~all_key
        :          ~~all
        """
    )
    root, _ = load_with_anchors(content)
    files_key = find_yaml_node_for_span(root, spans["files_key"])
    files = find_yaml_node_for_span(root, spans["files"])
    root_context = Mock()
    manager = MagicMock()

    expected_calls = [
        call.handler.handle_files(root_context, files_key, files),
        call.handler.handle_files().__enter__(),
        call.traverse_files_item(
            manager.handler,
            manager.handler.handle_files().__enter__(),
            find_yaml_node_for_span(root, spans["test_key"]),
            find_yaml_node_for_span(root, spans["test"]),
        ),
        call.traverse_files_item(
            manager.handler,
            manager.handler.handle_files().__enter__(),
            find_yaml_node_for_span(root, spans["all_key"]),
            find_yaml_node_for_span(root, spans["all"]),
        ),
        call.handler.handle_files().__exit__(None, None, None),
    ]
    manager.reset_mock()

    with patch(
        "rapids_pre_commit_hooks.utils.dependencies_yaml.traverse_files_item",
        manager.traverse_files_item,
    ):
        dependencies_yaml.traverse_files(
            manager.handler, root_context, files_key, files
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
    composed, _ = load_with_anchors(content)
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
            :       ~~~~items.0
            :     >packages
            +     - lib2
            :       ~~~~items.1
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
            :       ~~~~items.0
            +     - lib2
            :       ~~~~items.1
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
            :       ~~~~items.0
            +     - lib2
            :       ~~~~items.1
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
    composed, _ = load_with_anchors(content)
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
            find_yaml_node_for_span(composed, spans["items"][0]),
        ),
        call.traverse_package(
            manager.handler,
            manager.handler.handle_packages().__enter__(),
            anchors,
            used_anchors_after,
            find_yaml_node_for_span(composed, spans["items"][1]),
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
    content, spans = parse_named_spans(
        """\
        + [requirements]
        :  ~~~~~~~~~~~~output_type
        """
    )
    output_types, _ = load_with_anchors(content)
    output_type = find_yaml_node_for_span(output_types, spans["output_type"])
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
    item, _ = load_with_anchors(content)
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
    content, spans = parse_named_spans(
        """\
        + - output_types: pyproject
        :   >common_item
        :   ~~~~~~~~~~~~output_types_key
        :                 ~~~~~~~~~output_types
        +   packages: []
        :                !common_item
        :   ~~~~~~~~packages_key
        :             ~~packages
        """
    )
    common, _ = load_with_anchors(content)
    common_item = find_yaml_node_for_span(common, spans["common_item"])
    common_context = Mock()
    manager = MagicMock()

    expected_calls = [
        call.handler.handle_common_item(common_context, common_item),
        call.handler.handle_common_item().__enter__(),
        call.traverse_output_types(
            manager.handler,
            manager.handler.handle_common_item().__enter__(),
            find_yaml_node_for_span(common, spans["output_types_key"]),
            find_yaml_node_for_span(common, spans["output_types"]),
        ),
        call.traverse_packages(
            manager.handler,
            manager.handler.handle_common_item().__enter__(),
            {},
            set(),
            find_yaml_node_for_span(common, spans["packages_key"]),
            find_yaml_node_for_span(common, spans["packages"]),
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
    content, spans = parse_named_spans(
        """\
        + common:
        : ~~~~~~common_key
        +     - {}
        :     >common
        :       ~~common_item_0
        +     - {}
        :          !common
        :       ~~common_item_1
        """
    )
    dependency_set, _ = load_with_anchors(content)
    common_key = find_yaml_node_for_span(dependency_set, spans["common_key"])
    common = find_yaml_node_for_span(dependency_set, spans["common"])
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
            find_yaml_node_for_span(dependency_set, spans["common_item_0"]),
        ),
        call.traverse_common_item(
            manager.handler,
            manager.handler.handle_common().__enter__(),
            {},
            set(),
            find_yaml_node_for_span(dependency_set, spans["common_item_1"]),
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
    content, spans = parse_named_spans(
        """\
        + value_1: "true"
        : ~~~~~~~matrix_item_key
        :          ~~~~~~matrix_item
        """
    )
    matrix, _ = load_with_anchors(content)
    matrix_item_key = find_yaml_node_for_span(matrix, spans["matrix_item_key"])
    matrix_item = find_yaml_node_for_span(matrix, spans["matrix_item"])
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
    content, spans = parse_named_spans(
        """\
        + matrix:
        : ~~~~~~matrix_key
        +     value_1: "true"
        :     >matrix
        :     ~~~~~~~value_1_key
        :              ~~~~~~value_1
        +     value_2: "true"
        :                     !matrix
        :     ~~~~~~~value_2_key
        :              ~~~~~~value_2
        """
    )
    matrices_item, _ = load_with_anchors(content)
    matrix_key = find_yaml_node_for_span(matrices_item, spans["matrix_key"])
    matrix = find_yaml_node_for_span(matrices_item, spans["matrix"])
    matrices_item_context = Mock()
    manager = MagicMock()

    expected_calls = [
        call.handler.handle_matrix(matrices_item_context, matrix_key, matrix),
        call.handler.handle_matrix().__enter__(),
        call.traverse_matrix_item(
            manager.handler,
            manager.handler.handle_matrix().__enter__(),
            find_yaml_node_for_span(matrices_item, spans["value_1_key"]),
            find_yaml_node_for_span(matrices_item, spans["value_1"]),
        ),
        call.traverse_matrix_item(
            manager.handler,
            manager.handler.handle_matrix().__enter__(),
            find_yaml_node_for_span(matrices_item, spans["value_2_key"]),
            find_yaml_node_for_span(matrices_item, spans["value_2"]),
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
    content, spans = parse_named_spans(
        """\
        + - matrix: {}
        :   >matrices_item
        :   ~~~~~~matrix_key
        :           ~~matrix
        +   packages: []
        :                !matrices_item
        :   ~~~~~~~~packages_key
        :             ~~packages
        """
    )
    matrices, _ = load_with_anchors(content)
    matrices_item = find_yaml_node_for_span(matrices, spans["matrices_item"])
    matrices_context = Mock()
    manager = MagicMock()

    expected_calls = [
        call.handler.handle_matrices_item(matrices_context, matrices_item),
        call.handler.handle_matrices_item().__enter__(),
        call.traverse_matrix(
            manager.handler,
            manager.handler.handle_matrices_item().__enter__(),
            find_yaml_node_for_span(matrices, spans["matrix_key"]),
            find_yaml_node_for_span(matrices, spans["matrix"]),
        ),
        call.traverse_packages(
            manager.handler,
            manager.handler.handle_matrices_item().__enter__(),
            {},
            set(),
            find_yaml_node_for_span(matrices, spans["packages_key"]),
            find_yaml_node_for_span(matrices, spans["packages"]),
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
    content, spans = parse_named_spans(
        """\
        + matrices:
        : ~~~~~~~~matrices_key
        +     - {}
        :     >matrices
        :       ~~matrices_items.0
        +     - {}
        :       ~~matrices_items.1
        +     - {}
        :          !matrices
        :       ~~matrices_items.2
        """
    )
    specific_item, _ = load_with_anchors(content)
    matrices_key = find_yaml_node_for_span(
        specific_item, spans["matrices_key"]
    )
    matrices = find_yaml_node_for_span(specific_item, spans["matrices"])
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
            find_yaml_node_for_span(specific_item, spans["matrices_items"][0]),
        ),
        call.traverse_matrices_item(
            manager.handler,
            manager.handler.handle_matrices().__enter__(),
            {},
            set(),
            find_yaml_node_for_span(specific_item, spans["matrices_items"][1]),
        ),
        call.traverse_matrices_item(
            manager.handler,
            manager.handler.handle_matrices().__enter__(),
            {},
            set(),
            find_yaml_node_for_span(specific_item, spans["matrices_items"][2]),
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
    content, spans = parse_named_spans(
        """\
        + - output_types: pyproject
        :   >specific_item
        :   ~~~~~~~~~~~~output_types_key
        :                 ~~~~~~~~~output_types
        +   matrices: []
        :                !specific_item
        :   ~~~~~~~~matrices_key
        :             ~~matrices
        """
    )
    specific, _ = load_with_anchors(content)
    specific_item = find_yaml_node_for_span(specific, spans["specific_item"])
    specific_context = Mock()
    manager = MagicMock()

    expected_calls = [
        call.handler.handle_specific_item(specific_context, specific_item),
        call.handler.handle_specific_item().__enter__(),
        call.traverse_output_types(
            manager.handler,
            manager.handler.handle_specific_item().__enter__(),
            find_yaml_node_for_span(specific, spans["output_types_key"]),
            find_yaml_node_for_span(specific, spans["output_types"]),
        ),
        call.traverse_matrices(
            manager.handler,
            manager.handler.handle_specific_item().__enter__(),
            {},
            set(),
            find_yaml_node_for_span(specific, spans["matrices_key"]),
            find_yaml_node_for_span(specific, spans["matrices"]),
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
    content, spans = parse_named_spans(
        """\
        + specific:
        : ~~~~~~~~specific_key
        +     - {}
        :     >specific
        :       ~~specific_items.0
        +     - {}
        :       ~~specific_items.1
        +     - {}
        :          !specific
        :       ~~specific_items.2
        """
    )
    dependency_set, _ = load_with_anchors(content)
    specific_key = find_yaml_node_for_span(
        dependency_set, spans["specific_key"]
    )
    specific = find_yaml_node_for_span(dependency_set, spans["specific"])
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
            find_yaml_node_for_span(
                dependency_set, spans["specific_items"][0]
            ),
        ),
        call.traverse_specific_item(
            manager.handler,
            manager.handler.handle_specific().__enter__(),
            {},
            set(),
            find_yaml_node_for_span(
                dependency_set, spans["specific_items"][1]
            ),
        ),
        call.traverse_specific_item(
            manager.handler,
            manager.handler.handle_specific().__enter__(),
            {},
            set(),
            find_yaml_node_for_span(
                dependency_set, spans["specific_items"][2]
            ),
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
    content, spans = parse_named_spans(
        """\
        + dependency_set_1:
        : ~~~~~~~~~~~~~~~~dependency_set_key
        +     common: {}
        :     >dependency_set
        :     ~~~~~~common_key
        :             ~~common
        +     specific: {}
        :                  !dependency_set
        :     ~~~~~~~~specific_key
        :               ~~specific
        """
    )
    dependencies, _ = load_with_anchors(content)
    dependency_set_key = find_yaml_node_for_span(
        dependencies, spans["dependency_set_key"]
    )
    dependency_set = find_yaml_node_for_span(
        dependencies, spans["dependency_set"]
    )
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
            find_yaml_node_for_span(dependencies, spans["common_key"]),
            find_yaml_node_for_span(dependencies, spans["common"]),
        ),
        call.traverse_specific(
            manager.handler,
            manager.handler.handle_dependency_set().__enter__(),
            {},
            set(),
            find_yaml_node_for_span(dependencies, spans["specific_key"]),
            find_yaml_node_for_span(dependencies, spans["specific"]),
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
    content, spans = parse_named_spans(
        """\
        + dependencies:
        : ~~~~~~~~~~~~dependencies_key
        +     dependency_set_1: {}
        :     >dependencies
        :     ~~~~~~~~~~~~~~~~dependency_set_1_key
        :                       ~~dependency_set_1
        +     dependency_set_2: {}
        :                          !dependencies
        :     ~~~~~~~~~~~~~~~~dependency_set_2_key
        :                       ~~dependency_set_2
        """
    )
    root, _ = load_with_anchors(content)
    dependencies_key = find_yaml_node_for_span(root, spans["dependencies_key"])
    dependencies = find_yaml_node_for_span(root, spans["dependencies"])
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
            find_yaml_node_for_span(root, spans["dependency_set_1_key"]),
            find_yaml_node_for_span(root, spans["dependency_set_1"]),
        ),
        call.traverse_dependency_set(
            manager.handler,
            manager.handler.handle_dependencies().__enter__(),
            {},
            set(),
            find_yaml_node_for_span(root, spans["dependency_set_2_key"]),
            find_yaml_node_for_span(root, spans["dependency_set_2"]),
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
    content, spans = parse_named_spans(
        """\
        + files: {}
        : ~~~~~files_key
        :        ~~files
        + channels: []
        + dependencies: {}
        : ~~~~~~~~~~~~dependencies_key
        :               ~~dependencies
        """
    )
    root, _ = load_with_anchors(content)
    manager = MagicMock()

    expected_calls = [
        call.handler.handle_root(root),
        call.handler.handle_root().__enter__(),
        call.traverse_files(
            manager.handler,
            manager.handler.handle_root().__enter__(),
            find_yaml_node_for_span(root, spans["files_key"]),
            find_yaml_node_for_span(root, spans["files"]),
        ),
        call.traverse_dependencies(
            manager.handler,
            manager.handler.handle_root().__enter__(),
            {},
            set(),
            find_yaml_node_for_span(root, spans["dependencies_key"]),
            find_yaml_node_for_span(root, spans["dependencies"]),
        ),
        call.handler.handle_root().__exit__(None, None, None),
    ]
    manager.reset_mock()

    with (
        patch(
            "rapids_pre_commit_hooks.utils.dependencies_yaml.traverse_files",
            manager.traverse_files,
        ),
        patch(
            "rapids_pre_commit_hooks.utils.dependencies_yaml."
            "traverse_dependencies",
            manager.traverse_dependencies,
        ),
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
