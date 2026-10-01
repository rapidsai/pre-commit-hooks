# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock

import pytest

from rapids_pre_commit_hooks import lint
from rapids_pre_commit_hooks.dependencies.naming_conventions import (
    NamingConventionsHandler,
)
from rapids_pre_commit_hooks.utils import dependencies_yaml
from rapids_pre_commit_hooks_test_utils import (
    parse_named_spans,
    zip_expected_warnings,
)


def _compose(content):
    loader = dependencies_yaml.AnchorPreservingLoader(content)
    try:
        return loader.get_single_node()
    finally:
        loader.dispose()


class TestNamingConventionsHandler:
    @pytest.mark.parametrize(
        ["content", "expected_warnings"],
        [
            pytest.param(
                """\
                + files:
                +   wrong_name:
                :   ~~~~~~~~~~warnings.0.warning
                :   ~~~~~~~~~~warnings.0.replacements.0
                +     output: pyproject
                :             ~~~~~~~~~warnings.0.notes.0
                +     pyproject_dir: python/dask-cuda
                :                    ~~~~~~~~~~~~~~~~warnings.0.notes.1
                +     extras:
                +       table: project.optional-dependencies
                :              ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~warnings.0.notes.2
                +       key: test
                :            ~~~~warnings.0.notes.3
                """,
                [
                    {
                        "warning": "expected file key name is "
                        '"py_test_dask_cuda"',
                        "replacements": ["py_test_dask_cuda"],
                        "notes": [
                            "file key has pyproject output type",
                            'and project name "dask_cuda"',
                            'and table extra "project.optional-dependencies"',
                            'and key extra "test"',
                        ],
                    },
                ],
                id="extras-key",
            ),
            pytest.param(
                """\
                + files:
                +   wrong_name:
                :   ~~~~~~~~~~warnings.0.warning
                :   ~~~~~~~~~~warnings.0.replacements.0
                +     output: pyproject
                :             ~~~~~~~~~warnings.0.notes.0
                +     pyproject_dir: python/dask-cuda
                :                    ~~~~~~~~~~~~~~~~warnings.0.notes.1
                +     extras:
                +       table: project
                :              ~~~~~~~warnings.0.notes.2
                """,
                [
                    {
                        "warning": "expected file key name is "
                        '"py_run_dask_cuda"',
                        "replacements": ["py_run_dask_cuda"],
                        "notes": [
                            "file key has pyproject output type",
                            'and project name "dask_cuda"',
                            'and table extra "project"',
                        ],
                    },
                ],
                id="no-extras-key",
            ),
            pytest.param(
                """\
                + files:
                +   wrong_name:
                +     output: pyproject
                +     pyproject_dir: python/dask-cuda
                +     extras:
                +       table: unknown
                """,
                [],
                id="mismatched-table",
            ),
            pytest.param(
                """\
                + files:
                +   wrong_name:
                +     output: pyproject
                +     pyproject_dir: python/dask-cuda
                +     extras:
                +       table: project.optional-dependencies
                +       key: docs
                """,
                [],
                id="mismatched-key",
            ),
            pytest.param(
                """\
                + files:
                +   wrong_name:
                +     output: requirements
                +     pyproject_dir: python/dask-cuda
                +     extras:
                +       table: project
                """,
                [],
                id="non-pyproject-output",
            ),
            pytest.param(
                """\
                + files:
                +   wrong_name:
                +     output: pyproject
                +     pyproject_dir: python/
                +     extras:
                +       table: project
                """,
                [],
                id="empty-python-dir",
            ),
            pytest.param(
                """\
                + files:
                +   wrong_name:
                +     output: pyproject
                +     extras:
                +       table: project
                """,
                [],
                id="no-pyproject-dir",
            ),
            pytest.param(
                """\
                + files:
                +   wrong_name:
                +     output: pyproject
                +     pyproject_dir: python/dask-cuda
                """,
                [],
                id="no-extras-table",
            ),
            pytest.param(
                """\
                + files:
                +   py_run_dask_cuda:
                +     output: pyproject
                +     pyproject_dir: python/dask-cuda
                +     extras:
                +       table: project
                """,
                [],
                id="correct-naming-convention-without-key",
            ),
            pytest.param(
                """\
                + files:
                +   py_test_dask_cuda:
                +     output: pyproject
                +     pyproject_dir: python/dask-cuda
                +     extras:
                +       table: project.optional-dependencies
                +       key: test
                """,
                [],
                id="correct-naming-convention-with-key",
            ),
        ],
    )
    def test_handle_files_item(self, content, expected_warnings):
        content, spans = parse_named_spans(content, dict)
        root = _compose(content)
        files = root.value[0][1]
        file_key, file_value = files.value[0]
        file_fields = {key.value: value for key, value in file_value.value}
        extras_fields = (
            {key.value: value for key, value in file_fields["extras"].value}
            if "extras" in file_fields
            else {}
        )
        linter = lint.Linter(
            "dependencies.yaml", content, "verify-dependencies"
        )
        handler = NamingConventionsHandler(linter, Mock())

        with handler.handle_files_item(None, file_key, file_value) as context:
            handler.handle_file_output_item(context, file_fields["output"])
            if "pyproject_dir" in file_fields:
                handler.handle_pyproject_dir(
                    context, Mock(), file_fields["pyproject_dir"]
                )
            if "table" in extras_fields:
                handler.handle_extras_table(
                    context, Mock(), extras_fields["table"]
                )
            if "key" in extras_fields:
                handler.handle_extras_key(
                    context, Mock(), extras_fields["key"]
                )

        assert linter.warnings == zip_expected_warnings(
            spans.get("warnings", []),
            expected_warnings,
        )

    @pytest.mark.parametrize(
        ["output_type", "expected"],
        [
            pytest.param("pyproject", True, id="pyproject"),
            pytest.param("requirements", False, id="requirements"),
            pytest.param("conda", False, id="conda"),
            pytest.param("none", False, id="none"),
        ],
    )
    def test_handle_file_output_item(self, output_type, expected):
        handler = NamingConventionsHandler(Mock(), Mock())
        context = NamingConventionsHandler.FileContext()
        item = Mock(value=output_type)

        handler.handle_file_output_item(context, item)

        assert context.pyproject_output_node is (item if expected else None)

    @pytest.mark.parametrize(
        ["pyproject_dir", "project_name"],
        [
            pytest.param("python/cudf", "cudf", id="project"),
            pytest.param(
                "python/dask-cuda", "dask_cuda", id="hyphenated-project"
            ),
            pytest.param("python/cudf/subdir", None, id="nested-dir"),
            pytest.param("cpp/cudf", None, id="non-python-dir"),
        ],
    )
    def test_handle_pyproject_dir(self, pyproject_dir, project_name):
        handler = NamingConventionsHandler(Mock(), Mock())
        context = NamingConventionsHandler.FileContext()
        value = Mock(value=pyproject_dir)

        handler.handle_pyproject_dir(context, Mock(), value)

        assert context.project_name == project_name
        assert context.pyproject_dir_node is (
            value if project_name is not None else None
        )

    def test_handle_extras_table(self):
        handler = NamingConventionsHandler(Mock(), Mock())
        context = NamingConventionsHandler.FileContext()
        value = Mock(value="project")

        handler.handle_extras_table(context, Mock(), value)

        assert context.table_node is value

    def test_handle_extras_key(self):
        handler = NamingConventionsHandler(Mock(), Mock())
        context = NamingConventionsHandler.FileContext()
        value = Mock(value="test")

        handler.handle_extras_key(context, Mock(), value)

        assert context.key_node is value


@pytest.mark.parametrize(
    ["content", "warnings"],
    [
        pytest.param(
            """\
            + files:
            +   wrong_test_name:
            :   ~~~~~~~~~~~~~~~warnings.0.warning
            :   ~~~~~~~~~~~~~~~warnings.0.replacements.0
            +     output: [conda, pyproject]
            :                     ~~~~~~~~~warnings.0.notes.0
            +     pyproject_dir: python/dask-cuda
            :                    ~~~~~~~~~~~~~~~~warnings.0.notes.1
            +     extras:
            +       table: project.optional-dependencies
            :              ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~warnings.0.notes.2
            +       key: test
            :            ~~~~warnings.0.notes.3
            +   wrong_build_name:
            :   ~~~~~~~~~~~~~~~~warnings.1.warning
            :   ~~~~~~~~~~~~~~~~warnings.1.replacements.0
            +     output: pyproject
            :             ~~~~~~~~~warnings.1.notes.0
            +     pyproject_dir: python/cudf
            :                    ~~~~~~~~~~~warnings.1.notes.1
            +     extras:
            +       table: build-system
            :              ~~~~~~~~~~~~warnings.1.notes.2
            +   py_run_dask_cuda:
            +     output: pyproject
            +     pyproject_dir: python/dask-cuda
            +     extras:
            +       table: project
            +   requirements:
            +     output: requirements
            +     pyproject_dir: python/cudf
            +     extras:
            +       table: project
            + dependencies: {}
            """,
            [
                {
                    "warning": 'expected file key name is "py_test_dask_cuda"',
                    "replacements": ["py_test_dask_cuda"],
                    "notes": [
                        "file key has pyproject output type",
                        'and project name "dask_cuda"',
                        'and table extra "project.optional-dependencies"',
                        'and key extra "test"',
                    ],
                },
                {
                    "warning": 'expected file key name is "py_build_cudf"',
                    "replacements": ["py_build_cudf"],
                    "notes": [
                        "file key has pyproject output type",
                        'and project name "cudf"',
                        'and table extra "build-system"',
                    ],
                },
            ],
            id="files",
        ),
    ],
)
def test_check_naming_conventions_integration(content, warnings):
    content, spans = parse_named_spans(content, dict)

    loader = dependencies_yaml.AnchorPreservingLoader(content)
    try:
        composed = loader.get_single_node()
    finally:
        loader.dispose()

    args = Mock()
    linter = lint.Linter("dependencies.yaml", content, "verify-dependencies")
    handler = NamingConventionsHandler(linter, args)

    dependencies_yaml.traverse_root(
        handler, loader.document_anchors[0], set(), composed
    )

    assert linter.warnings == zip_expected_warnings(
        spans.get("warnings", []), warnings
    )
