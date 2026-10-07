# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from rapids_pre_commit_hooks import lint
from rapids_pre_commit_hooks.dependencies.cuda_suffixed import (
    CUDASuffixedHandler,
)
from rapids_pre_commit_hooks.utils import dependencies_yaml
from rapids_pre_commit_hooks.utils.yaml import (
    Anchor,
    AnchorType,
    load_with_anchors,
)
from rapids_pre_commit_hooks_test_utils import (
    find_yaml_node_for_span,
    parse_named_spans,
    zip_expected_warnings,
)


class TestCUDASuffixedHandler:
    @pytest.mark.parametrize(
        ["output_type", "expected"],
        [
            pytest.param("requirements", True, id="requirements"),
            pytest.param("constraints", True, id="constraints"),
            pytest.param("pyproject", True, id="pyproject"),
            pytest.param("conda", False, id="conda"),
        ],
    )
    def test_handle_output_type(self, output_type, expected):
        handler = CUDASuffixedHandler(Mock(), Mock())
        context = CUDASuffixedHandler.CommonItemContext()

        handler.handle_output_type(context, Mock(value=output_type))
        assert context.has_python_output_type is expected

        context.has_python_output_type = True
        handler.handle_output_type(context, Mock(value=output_type))
        assert context.has_python_output_type is True

    def test_handle_common(self):
        composed, _ = load_with_anchors(
            """\
            common:
              packages: []
            """
        )
        common_key, common = composed.value[0]

        handler = CUDASuffixedHandler(Mock(), Mock())
        with handler.handle_common(Mock(), common_key, common) as context:
            assert context.common_key == common_key

    @pytest.mark.parametrize(
        [
            "content",
            "has_python_output_type",
            "suffixed_names",
            "unsuffixed_names",
            "warnings",
        ],
        [
            pytest.param(
                """\
                + common:
                : ~~~~~~warnings.0.notes.0
                : ~~~~~~warnings.1.notes.0
                +   packages:
                +     - package-cu12
                :       ~~~~~~~~~~~~suffixed.0
                :       ~~~~~~~~~~~~warnings.0.warning
                +     - package
                :       ~~~~~~~unsuffixed.0
                :       ~~~~~~~warnings.1.warning
                """,
                True,
                ["package"],
                ["package"],
                [
                    {
                        "warning": 'package "package" in common '
                        "dependency set",
                        "notes": [
                            "place in a specific dependency set with "
                            'cuda_suffixed: "true" instead',
                        ],
                    },
                    {
                        "warning": 'package "package" in common '
                        "dependency set",
                        "notes": [
                            "place in a specific dependency set with "
                            'cuda_suffixed: "false" instead',
                        ],
                    },
                ],
                id="both-package-forms",
            ),
            pytest.param(
                """\
                + common:
                +   packages:
                +     - package-cu12
                :       ~~~~~~~~~~~~suffixed.0
                +     - package
                :       ~~~~~~~unsuffixed.0
                """,
                False,
                ["package"],
                ["package"],
                [],
                id="non-python-output",
            ),
            pytest.param(
                """\
                + common:
                +   packages: []
                """,
                True,
                [],
                [],
                [],
                id="no-suspicious-packages",
            ),
        ],
    )
    def test_handle_common_item(
        self,
        content,
        has_python_output_type,
        suffixed_names,
        unsuffixed_names,
        warnings,
    ):
        content, spans = parse_named_spans(content, dict)
        composed, _ = load_with_anchors(content)
        common_key, common = composed.value[0]
        linter = lint.Linter(
            "dependencies.yaml", content, "verify-dependencies"
        )
        handler = CUDASuffixedHandler(linter, Mock())

        common_context = Mock(common_key=common_key)
        with handler.handle_common_item(
            common_context, common
        ) as item_context:
            item_context.has_python_output_type = has_python_output_type
            item_context.suspicious_suffixed_packages.extend(
                (
                    name,
                    None,
                    None,
                    find_yaml_node_for_span(composed, span),
                )
                for name, span in zip(
                    suffixed_names,
                    spans.get("suffixed", []),
                    strict=True,
                )
            )
            item_context.suspicious_unsuffixed_packages.extend(
                (
                    name,
                    None,
                    find_yaml_node_for_span(composed, span),
                )
                for name, span in zip(
                    unsuffixed_names,
                    spans.get("unsuffixed", []),
                    strict=True,
                )
            )

        assert linter.warnings == zip_expected_warnings(
            spans.get("warnings", []), warnings
        )

    @pytest.mark.parametrize(
        [
            "content",
            "has_python_output_type",
            "cuda_suffixed",
            "cuda_major",
            "suffixed_names",
            "unsuffixed_names",
            "warnings",
        ],
        [
            pytest.param(
                """\
                + matrix:
                : ~~~~~~matrix
                : ~~~~~~warnings.0.notes.0
                +   cuda: "12.0"
                + packages:
                +   - package-cu12
                :     ~~~~~~~~~~~~suffixed.0
                :     ~~~~~~~~~~~~warnings.0.warning
                """,
                True,
                None,
                None,
                [("package", "-cu12", None)],
                [],
                [
                    {
                        "warning": 'package "package" in specific dependency '
                        "set with no cuda_suffixed field",
                        "notes": [
                            "place in a specific dependency set with "
                            'cuda_suffixed: "true" instead',
                        ],
                    },
                ],
                id="no-field-suffixed-package",
            ),
            pytest.param(
                """\
                + matrix:
                : ~~~~~~matrix
                : ~~~~~~warnings.0.notes.0
                +   cuda: "12.0"
                + packages:
                +   - package
                :     ~~~~~~~unsuffixed.0
                :     ~~~~~~~warnings.0.warning
                """,
                True,
                None,
                None,
                [],
                [("package", None)],
                [
                    {
                        "warning": 'package "package" in specific '
                        "dependency set with no cuda_suffixed field",
                        "notes": [
                            "place in a specific dependency set with "
                            'cuda_suffixed: "false" instead',
                        ],
                    },
                ],
                id="no-field-unsuffixed-package",
            ),
            pytest.param(
                """\
                + matrix:
                : ~~~~~~matrix
                : ~~~~~~warnings.0.notes.0
                +   cuda_suffixed: "true"
                + packages:
                +   - package
                :     ~~~~~~~unsuffixed.0
                :     ~~~~~~~warnings.0.warning
                """,
                True,
                True,
                None,
                [],
                [("package", None)],
                [
                    {
                        "warning": 'package "package" in specific dependency '
                        'set with cuda_suffixed: "true"',
                        "notes": [
                            "add a cuda matrix field and add matching -cu* "
                            "suffix to package name",
                        ],
                    },
                ],
                id="true-unsuffixed-package",
            ),
            pytest.param(
                """\
                + matrix:
                : ~~~~~~matrix
                +   cuda_suffixed: "true"
                +   cuda: "12.8"
                + packages:
                +   - package
                :     ~~~~~~~unsuffixed.0
                :     ~~~~~~~warnings.0.warning
                :     ~~~~~~~warnings.0.replacements.0
                """,
                True,
                True,
                12,
                [],
                [("package", None)],
                [
                    {
                        "warning": 'package "package" in specific dependency '
                        'set with cuda_suffixed: "true"',
                        "replacements": [
                            "package-cu12",
                        ],
                    },
                ],
                id="true-unsuffixed-package-cuda-major",
            ),
            pytest.param(
                """\
                + matrix:
                : ~~~~~~matrix
                +   cuda_suffixed: "true"
                +   cuda: "12.8"
                + packages:
                +   - package==26.08.*,>=0.0.0a0
                :     ~~~~~~~~~~~~~~~~~~~~~~~~~~unsuffixed.0
                :     ~~~~~~~~~~~~~~~~~~~~~~~~~~warnings.0.warning
                :     ~~~~~~~~~~~~~~~~~~~~~~~~~~warnings.0.replacements.0
                """,
                True,
                True,
                12,
                [],
                [("package", None)],
                [
                    {
                        "warning": 'package "package" in specific dependency '
                        'set with cuda_suffixed: "true"',
                        "replacements": [
                            "package-cu12==26.08.*,>=0.0.0a0",
                        ],
                    },
                ],
                id="true-unsuffixed-package-cuda-major-version-req",
            ),
            pytest.param(
                """\
                + matrix:
                : ~~~~~~matrix
                +   cuda_suffixed: "true"
                +   cuda: "12.8"
                + packages:
                +   - &package_anchor package
                :     ~~~~~~~~~~~~~~~~~~~~~~~unsuffixed.0
                :     ~~~~~~~~~~~~~~~~~~~~~~~warnings.0.warning
                :     ~~~~~~~~~~~~~~~~~~~~~~~warnings.0.replacements.0
                """,
                True,
                True,
                12,
                [],
                [
                    (
                        "package",
                        Anchor(AnchorType.DEFINITION, "package_anchor"),
                    )
                ],
                [
                    {
                        "warning": 'package "package" in specific dependency '
                        'set with cuda_suffixed: "true"',
                        "replacements": [
                            "&package_anchor package-cu12",
                        ],
                    },
                ],
                id="true-unsuffixed-package-cuda-major-anchor",
            ),
            pytest.param(
                """\
                + matrix:
                +   cuda_suffixed: "true"
                + packages:
                +   - package-cu12
                :     ~~~~~~~~~~~~suffixed.0
                """,
                True,
                True,
                None,
                [("package", "-cu12", None)],
                [],
                [],
                id="true-suffixed-package",
            ),
            pytest.param(
                """\
                + matrix:
                +   cuda_suffixed: "true"
                +   cuda: "12.*"
                + packages:
                +   - package-cu12
                :     ~~~~~~~~~~~~suffixed.0
                """,
                True,
                True,
                12,
                [("package", "-cu12", None)],
                [],
                [],
                id="true-suffixed-package-cuda-major",
            ),
            pytest.param(
                """\
                + matrix:
                +   cuda_suffixed: "true"
                +   cuda: "13.*"
                + packages:
                +   - package-cu12
                :     ~~~~~~~~~~~~suffixed.0
                :     ~~~~~~~~~~~~warnings.0.warning
                :     ~~~~~~~~~~~~warnings.0.replacements.0
                """,
                True,
                True,
                13,
                [("package", "-cu12", None)],
                [],
                [
                    {
                        "warning": 'package "package" has wrong -cu* suffix',
                        "replacements": [
                            "package-cu13",
                        ],
                    },
                ],
                id="true-suffixed-package-wrong-cuda-major",
            ),
            pytest.param(
                """\
                + matrix:
                +   cuda_suffixed: "true"
                +   cuda: "13.*"
                + packages:
                +   - package-cu12==26.08.*,>=0.0.0a0
                :     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~suffixed.0
                :     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~warnings.0.warning
                :     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~warnings.0.replacements.0
                """,
                True,
                True,
                13,
                [("package", "-cu12", None)],
                [],
                [
                    {
                        "warning": 'package "package" has wrong -cu* suffix',
                        "replacements": [
                            "package-cu13==26.08.*,>=0.0.0a0",
                        ],
                    },
                ],
                id="true-suffixed-package-wrong-cuda-major-version-req",
            ),
            pytest.param(
                """\
                + matrix:
                +   cuda_suffixed: "true"
                +   cuda: "13.*"
                + packages:
                +   - &package_anchor package-cu12
                :     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~suffixed.0
                :     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~warnings.0.warning
                :     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~warnings.0.replacements.0
                """,
                True,
                True,
                13,
                [
                    (
                        "package",
                        "-cu12",
                        Anchor(AnchorType.DEFINITION, "package_anchor"),
                    )
                ],
                [],
                [
                    {
                        "warning": 'package "package" has wrong -cu* suffix',
                        "replacements": [
                            "&package_anchor package-cu13",
                        ],
                    },
                ],
                id="true-suffixed-package-wrong-cuda-major-anchor",
            ),
            pytest.param(
                """\
                + matrix:
                : ~~~~~~matrix
                +   cuda_suffixed: "false"
                + packages:
                +   - package-cu12
                :     ~~~~~~~~~~~~suffixed.0
                :     ~~~~~~~~~~~~warnings.0.warning
                :     ~~~~~~~~~~~~warnings.0.replacements.0
                """,
                True,
                False,
                None,
                [("package", "-cu12", None)],
                [],
                [
                    {
                        "warning": 'package "package" in specific dependency '
                        'set with cuda_suffixed: "false"',
                        "replacements": [
                            "package",
                        ],
                    },
                ],
                id="false-suffixed-package",
            ),
            pytest.param(
                """\
                + matrix:
                : ~~~~~~matrix
                +   cuda_suffixed: "false"
                + packages:
                +   - package-cu12==26.08.*,>=0.0.0a0
                :     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~suffixed.0
                :     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~warnings.0.warning
                :     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~warnings.0.replacements.0
                """,
                True,
                False,
                None,
                [("package", "-cu12", None)],
                [],
                [
                    {
                        "warning": 'package "package" in specific dependency '
                        'set with cuda_suffixed: "false"',
                        "replacements": [
                            "package==26.08.*,>=0.0.0a0",
                        ],
                    },
                ],
                id="false-suffixed-package-version-req",
            ),
            pytest.param(
                """\
                + matrix:
                : ~~~~~~matrix
                +   cuda_suffixed: "false"
                + packages:
                +   - &package_anchor package-cu12
                :     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~suffixed.0
                :     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~warnings.0.warning
                :     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~warnings.0.replacements.0
                """,
                True,
                False,
                None,
                [
                    (
                        "package",
                        "-cu12",
                        Anchor(AnchorType.DEFINITION, "package_anchor"),
                    )
                ],
                [],
                [
                    {
                        "warning": 'package "package" in specific dependency '
                        'set with cuda_suffixed: "false"',
                        "replacements": [
                            "&package_anchor package",
                        ],
                    },
                ],
                id="false-suffixed-package-anchor",
            ),
            pytest.param(
                """\
                + matrix:
                +   cuda_suffixed: "false"
                + packages:
                +   - package
                :     ~~~~~~~unsuffixed.0
                """,
                True,
                False,
                None,
                [],
                [("package", None)],
                [],
                id="false-unsuffixed-package",
            ),
            pytest.param(
                """\
                + packages:
                +   - package-cu12
                :     ~~~~~~~~~~~~suffixed.0
                """,
                False,
                None,
                None,
                [("package", "-cu12", None)],
                [],
                [],
                id="non-python-output",
            ),
        ],
    )
    def test_handle_specific_item(
        self,
        content,
        has_python_output_type,
        cuda_suffixed,
        cuda_major,
        suffixed_names,
        unsuffixed_names,
        warnings,
    ):
        content, spans = parse_named_spans(content, dict)
        composed, _ = load_with_anchors(content)
        linter = lint.Linter(
            "dependencies.yaml", content, "verify-dependencies"
        )
        handler = CUDASuffixedHandler(linter, Mock())
        matrix_node = (
            find_yaml_node_for_span(composed, span)
            if (span := spans.get("matrix"))
            else None
        )

        with handler.handle_specific_item(
            Mock(), composed
        ) as specific_context:
            specific_context.has_python_output_type = has_python_output_type
            matrix_context = CUDASuffixedHandler.MatricesItemContext(
                matrix_node=matrix_node,
                cuda_suffixed=cuda_suffixed,
                cuda_major=cuda_major,
                suspicious_suffixed_packages=[
                    (
                        name,
                        suffix,
                        anchor,
                        find_yaml_node_for_span(composed, span),
                    )
                    for (name, suffix, anchor), span in zip(
                        suffixed_names,
                        spans.get("suffixed", []),
                        strict=True,
                    )
                ],
                suspicious_unsuffixed_packages=[
                    (name, anchor, find_yaml_node_for_span(composed, span))
                    for (name, anchor), span in zip(
                        unsuffixed_names,
                        spans.get("unsuffixed", []),
                        strict=True,
                    )
                ],
            )
            specific_context.matrices_item_contexts.append(matrix_context)

        assert linter.warnings == zip_expected_warnings(
            spans.get("warnings", []), warnings
        )

    def test_handle_matrices_item(self):
        handler = CUDASuffixedHandler(Mock(), Mock())
        specific_context = CUDASuffixedHandler.SpecificItemContext()

        with handler.handle_matrices_item(
            specific_context, Mock()
        ) as matrix_context:
            assert specific_context.matrices_item_contexts == []

        assert specific_context.matrices_item_contexts == [matrix_context]

    def test_handle_matrix(self):
        content, spans = parse_named_spans(
            """\
            + matrix:
            : ~~~~~~matrix_key
            +   cuda_suffixed: "true"
            """
        )
        composed, _ = load_with_anchors(content)
        matrix_key, matrix = composed.value[0]
        context = CUDASuffixedHandler.MatricesItemContext()
        handler = CUDASuffixedHandler(Mock(), Mock())

        with handler.handle_matrix(
            context, matrix_key, matrix
        ) as matrix_context:
            assert matrix_context is context
            assert context.matrix_node == find_yaml_node_for_span(
                composed, spans["matrix_key"]
            )

    @pytest.mark.parametrize(
        [
            "content",
            "expected_cuda_suffixed",
            "has_cuda_suffixed_node",
            "expected_cuda_major",
            "has_cuda_node",
        ],
        [
            pytest.param(
                'cuda_suffixed: "true"',
                True,
                True,
                None,
                False,
                id="cuda-suffixed-true",
            ),
            pytest.param(
                'cuda_suffixed: "false"',
                False,
                True,
                None,
                False,
                id="cuda-suffixed-false",
            ),
            pytest.param(
                'cuda_suffixed: "other"',
                None,
                True,
                None,
                False,
                id="cuda-suffixed-other",
            ),
            pytest.param(
                'cuda: "12.8"',
                None,
                False,
                12,
                True,
                id="cuda-version",
            ),
            pytest.param(
                'cuda: "12.*"',
                None,
                False,
                12,
                True,
                id="cuda-version-wildcard",
            ),
            pytest.param(
                'cuda: "invalid"',
                None,
                False,
                None,
                False,
                id="cuda-version-invalid",
            ),
            pytest.param(
                'other: "value"',
                None,
                False,
                None,
                False,
                id="other",
            ),
        ],
    )
    def test_handle_matrix_item(
        self,
        content,
        expected_cuda_suffixed,
        has_cuda_suffixed_node,
        expected_cuda_major,
        has_cuda_node,
    ):
        composed, _ = load_with_anchors(content)
        key, value = composed.value[0]
        context = CUDASuffixedHandler.MatricesItemContext()
        handler = CUDASuffixedHandler(Mock(), Mock())

        handler.handle_matrix_item(context, key, value)

        assert context.cuda_suffixed is expected_cuda_suffixed
        assert context.cuda_suffixed_node == (
            value if has_cuda_suffixed_node else None
        )
        assert context.cuda_major == expected_cuda_major
        assert context.cuda_node == (value if has_cuda_node else None)

    @pytest.mark.parametrize(
        [
            "requirement",
            "anchor",
            "packages_is_reference_anchor",
            "suffixed_names",
            "unsuffixed_names",
        ],
        [
            pytest.param(
                "package",
                None,
                False,
                [],
                ["package"],
                id="unsuffixed",
            ),
            pytest.param(
                "package[extra]>=1.0",
                None,
                False,
                [],
                ["package"],
                id="unsuffixed-with-extras-and-version",
            ),
            pytest.param(
                "package-cu12",
                None,
                False,
                [("package", "-cu12")],
                [],
                id="suffixed",
            ),
            pytest.param(
                "package-cu123==1.0",
                None,
                False,
                [("package", "-cu123")],
                [],
                id="multi-digit-suffix",
            ),
            pytest.param(
                "package-cu12x",
                None,
                False,
                [],
                [],
                id="invalid-cuda-suffix",
            ),
            pytest.param(
                "other-cu12",
                None,
                False,
                [],
                [],
                id="unknown-package",
            ),
            pytest.param(
                "not a requirement",
                None,
                False,
                [],
                [],
                id="invalid-requirement",
            ),
            pytest.param(
                "package",
                Anchor(AnchorType.DEFINITION, "package"),
                False,
                [],
                ["package"],
                id="unsuffixed-anchor-definition",
            ),
            pytest.param(
                "package",
                Anchor(AnchorType.REFERENCE, "package"),
                False,
                [],
                ["package"],
                id="unsuffixed-anchor-reference",
            ),
            pytest.param(
                "package",
                None,
                True,
                [],
                [],
                id="unsuffixed-packages-is-reference-anchor",
            ),
            pytest.param(
                "package-cu12",
                Anchor(AnchorType.DEFINITION, "package"),
                False,
                [("package", "-cu12")],
                [],
                id="suffixed-anchor-definition",
            ),
            pytest.param(
                "package-cu12",
                Anchor(AnchorType.REFERENCE, "package"),
                False,
                [("package", "-cu12")],
                [],
                id="suffixed-anchor-reference",
            ),
            pytest.param(
                "package-cu12",
                None,
                True,
                [],
                [],
                id="suffixed-packages-is-reference-anchor",
            ),
        ],
    )
    def test_handle_package(
        self,
        requirement,
        anchor,
        packages_is_reference_anchor,
        suffixed_names,
        unsuffixed_names,
    ):
        package_node, _ = load_with_anchors(requirement)
        rapids_version = SimpleNamespace(cuda_suffixed_packages={"package"})
        context = CUDASuffixedHandler.PackagesContext(
            CUDASuffixedHandler.MatricesItemContext(),
            packages_is_reference_anchor,
        )
        handler = CUDASuffixedHandler(Mock(), Mock())

        with patch(
            "rapids_pre_commit_hooks.dependencies.cuda_suffixed."
            "get_rapids_version",
            return_value=rapids_version,
        ):
            handler.handle_package(context, anchor, package_node)

        assert context.parent_context.suspicious_suffixed_packages == [
            (
                name,
                suffix,
                anchor,
                package_node,
            )
            for (name, suffix) in suffixed_names
        ]
        assert context.parent_context.suspicious_unsuffixed_packages == [
            (name, anchor, package_node) for name in unsuffixed_names
        ]


@pytest.mark.parametrize(
    ["content", "warnings"],
    [
        pytest.param(
            """\
            + dependencies:
            +   file_set:
            +     common:
            :     ~~~~~~warnings.0.notes.0
            :     ~~~~~~warnings.1.notes.0
            +       - output_types: pyproject
            +         packages:
            +           - package-cu12
            :             ~~~~~~~~~~~~warnings.0.warning
            +           - package
            :             ~~~~~~~warnings.1.warning
            """,
            [
                {
                    "warning": 'package "package" in common dependency set',
                    "notes": [
                        "place in a specific dependency set with "
                        'cuda_suffixed: "true" instead',
                    ],
                },
                {
                    "warning": 'package "package" in common dependency set',
                    "notes": [
                        "place in a specific dependency set with "
                        'cuda_suffixed: "false" instead',
                    ],
                },
            ],
            id="common-python-packages",
        ),
        pytest.param(
            """\
            + dependencies:
            +   file_set:
            +     common:
            +       - output_types: conda
            +         packages:
            +           - package-cu12
            +           - package
            """,
            [],
            id="common-non-python-output",
        ),
        pytest.param(
            """\
            + dependencies:
            +   file_set:
            +     specific:
            +       - output_types: requirements
            +         matrices:
            +           - matrix:
            :             ~~~~~~warnings.0.notes.0
            +               cuda: "12.8"
            +             packages:
            +               - package-cu12
            :                 ~~~~~~~~~~~~warnings.0.warning
            +           - matrix:
            +               cuda_suffixed: "true"
            +               cuda: "12.8"
            +             packages:
            +               - package
            :                 ~~~~~~~warnings.1.warning
            :                 ~~~~~~~warnings.1.replacements.0
            +           - matrix:
            :             ~~~~~~warnings.2.notes.0
            +               cuda_suffixed: "true"
            +             packages:
            +               - package
            :                 ~~~~~~~warnings.2.warning
            +           - matrix:
            +               cuda_suffixed: "false"
            +             packages:
            +               - package-cu12
            :                 ~~~~~~~~~~~~warnings.3.warning
            :                 ~~~~~~~~~~~~warnings.3.replacements.0
            """,
            [
                {
                    "warning": 'package "package" in specific dependency set '
                    "with no cuda_suffixed field",
                    "notes": [
                        "place in a specific dependency set with "
                        'cuda_suffixed: "true" instead',
                    ],
                },
                {
                    "warning": 'package "package" in specific dependency set '
                    'with cuda_suffixed: "true"',
                    "replacements": [
                        "package-cu12",
                    ],
                },
                {
                    "warning": 'package "package" in specific dependency set '
                    'with cuda_suffixed: "true"',
                    "notes": [
                        "add a cuda matrix field and add matching -cu* "
                        "suffix to package name",
                    ],
                },
                {
                    "warning": 'package "package" in specific dependency set '
                    'with cuda_suffixed: "false"',
                    "replacements": [
                        "package",
                    ],
                },
            ],
            id="specific-invalid-package-forms",
        ),
        pytest.param(
            """\
            + dependencies:
            +   file_set:
            +     specific:
            +       - output_types: pyproject
            +         matrices:
            +           - matrix:
            +               cuda_suffixed: "true"
            +             packages:
            +               - package-cu12
            +           - matrix:
            +               cuda_suffixed: "false"
            +             packages:
            +               - package
            """,
            [],
            id="specific-valid-package-forms",
        ),
        pytest.param(
            """\
            + dependencies:
            +   file_set:
            +     common:
            +       - output_types: pyproject
            +         packages:
            +           - non-rapids-package
            +           - non-rapids-package-cu12
            +     specific:
            +       - output_types: pyproject
            +         matrices:
            +           - matrix:
            +             packages:
            +               - non-rapids-package
            +               - non-rapids-package-cu12
            +           - matrix:
            +               cuda_suffixed: "false"
            +             packages:
            +               - non-rapids-package
            +               - non-rapids-package-cu12
            +           - matrix:
            +               cuda_suffixed: "true"
            +             packages:
            +               - non-rapids-package
            +               - non-rapids-package-cu12
            """,
            [],
            id="non-rapids-packages",
        ),
        pytest.param(
            """\
            + dependencies:
            +   file_set:
            +     common:
            :     ~~~~~~warnings.0.notes.0
            :     ~~~~~~warnings.1.notes.0
            :     ~~~~~~warnings.2.notes.0
            :     ~~~~~~warnings.3.notes.0
            +       - output_types: pyproject
            +         packages:
            +           - &package package
            :             ~~~~~~~~~~~~~~~~warnings.1.warning
            :             ~~~~~~~~~~~~~~~~warnings.3.warning
            :             ~~~~~~~~~~~~~~~~warnings.4.warning
            +           - &package_cu12 package-cu12
            :             ~~~~~~~~~~~~~~~~~~~~~~~~~~warnings.0.warning
            :             ~~~~~~~~~~~~~~~~~~~~~~~~~~warnings.2.warning
            :             ~~~~~~~~~~~~~~~~~~~~~~~~~~warnings.5.warning
            +       - output_types: pyproject
            +         packages:
            +           - *package
            +           - *package_cu12
            +     specific:
            +       - output_types: pyproject
            +         matrices:
            +           - matrix:
            :             ~~~~~~warnings.4.notes.0
            +               cuda_suffixed: "true"
            +               cuda: "12.*"
            +             packages:
            +               - *package
            +           - matrix:
            :             ~~~~~~warnings.5.notes.0
            +               cuda_suffixed: "false"
            +             packages:
            +               - *package_cu12
            """,
            [
                {
                    "warning": 'package "package" in common dependency set',
                    "notes": [
                        "place in a specific dependency set with "
                        'cuda_suffixed: "true" instead',
                    ],
                },
                {
                    "warning": 'package "package" in common dependency set',
                    "notes": [
                        "place in a specific dependency set with "
                        'cuda_suffixed: "false" instead',
                    ],
                },
                {
                    "warning": 'package "package" in common dependency set',
                    "notes": [
                        "place in a specific dependency set with "
                        'cuda_suffixed: "true" instead',
                    ],
                },
                {
                    "warning": 'package "package" in common dependency set',
                    "notes": [
                        "place in a specific dependency set with "
                        'cuda_suffixed: "false" instead',
                    ],
                },
                {
                    "warning": 'package "package" in specific dependency set '
                    'with cuda_suffixed: "true"',
                    "notes": [
                        "place in a specific dependency set with "
                        'cuda_suffixed: "false" instead',
                    ],
                },
                {
                    "warning": 'package "package" in specific dependency set '
                    'with cuda_suffixed: "false"',
                    "notes": [
                        "place in a specific dependency set with "
                        'cuda_suffixed: "true" instead',
                    ],
                },
            ],
            id="anchors",
        ),
    ],
)
def test_check_cuda_suffixed_integration(content, warnings):
    content, spans = parse_named_spans(content, dict)

    composed, anchors = load_with_anchors(content)

    args = Mock()
    linter = lint.Linter("dependencies.yaml", content, "verify-dependencies")
    rapids_version = SimpleNamespace(cuda_suffixed_packages={"package"})

    handler = CUDASuffixedHandler(linter, args)

    with patch(
        "rapids_pre_commit_hooks.dependencies.cuda_suffixed."
        "get_rapids_version",
        return_value=rapids_version,
    ):
        dependencies_yaml.traverse_root(handler, anchors, set(), composed)

    assert linter.warnings == zip_expected_warnings(
        spans.get("warnings", []), warnings
    )
