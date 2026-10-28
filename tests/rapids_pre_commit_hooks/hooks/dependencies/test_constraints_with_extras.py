# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import pytest

from rapids_pre_commit_hooks.hooks.dependencies.constraints_with_extras import (  # noqa: E501
    ConstraintsWithExtrasHandler,
)
from rapids_pre_commit_hooks import Linter
from rapids_pre_commit_hooks.utils.dependencies_yaml import (
    traverse_dependencies_yaml,
)
from rapids_pre_commit_hooks_test_utils import (
    parse_named_spans,
    zip_expected_warnings,
)


@pytest.mark.parametrize(
    ["content", "warnings"],
    [
        pytest.param(
            """\
            + dependencies:
            +   test:
            +     common:
            +       - packages:
            +           - 'package[foo,bar]>=1; python_version >= "3.12"'
            :             ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~warnings.0.warning
            :             ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~warnings.0.replacements.0
            +           - --extra-index-url=https://example.com/[extra]
            +           - not a requirement
            +           - other>=2
            +         output_types: constraints
            :                       ~~~~~~~~~~~warnings.0.notes.0
            """,  # noqa: E501
            [
                {
                    "warning": (
                        'package "package" has extras in a constraints '
                        "dependency set"
                    ),
                    "replacements": [
                        ("'package>=1; python_version >= \"3.12\"'"),
                    ],
                    "notes": [
                        (
                            "the constraints output type does not support "
                            "extras"
                        ),
                    ],
                },
            ],
            id="scalar-constraints",
        ),
        pytest.param(
            """\
            + dependencies:
            +   test:
            +     common:
            +       - packages:
            +           - 'package[foo,bar]>=1; python_version >= "3.12"'
            :             ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~warnings.0.warning
            :             ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~warnings.0.replacements.0
            +           - --extra-index-url=https://example.com/[extra]
            +           - not a requirement
            +           - other>=2
            +         output_types: [constraints]
            :                        ~~~~~~~~~~~warnings.0.notes.0
            """,  # noqa: E501
            [
                {
                    "warning": (
                        'package "package" has extras in a constraints '
                        "dependency set"
                    ),
                    "replacements": [
                        ("'package>=1; python_version >= \"3.12\"'"),
                    ],
                    "notes": [
                        (
                            "the constraints output type does not support "
                            "extras"
                        ),
                    ],
                },
            ],
            id="list-constraints",
        ),
        pytest.param(
            """\
            + dependencies:
            +   test:
            +     common:
            +       - packages:
            +           - 'package[foo,bar]>=1; python_version >= "3.12"'
            :             ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~warnings.0.warning
            +           - --extra-index-url=https://example.com/[extra]
            +           - not a requirement
            +           - other>=2
            +         output_types: [constraints, requirements]
            :                        ~~~~~~~~~~~warnings.0.notes.0
            :                                     ~~~~~~~~~~~~warnings.0.notes.1
            """,  # noqa: E501
            [
                {
                    "warning": (
                        'package "package" has extras in a constraints '
                        "dependency set"
                    ),
                    "notes": [
                        (
                            "the constraints output type does not support "
                            "extras"
                        ),
                        (
                            "separate the constraints and requirements "
                            "dependency lists to preserve extras"
                        ),
                    ],
                },
            ],
            id="mixed-requirements",
        ),
        pytest.param(
            """\
            + dependencies:
            +   test:
            +     common:
            +       - packages:
            +           - 'package[foo,bar]>=1; python_version >= "3.12"'
            :             ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~warnings.0.warning
            +           - --extra-index-url=https://example.com/[extra]
            +           - not a requirement
            +           - other>=2
            +         output_types: [conda, constraints]
            :                               ~~~~~~~~~~~warnings.0.notes.0
            :                        ~~~~~warnings.0.notes.1
            """,  # noqa: E501
            [
                {
                    "warning": (
                        'package "package" has extras in a constraints '
                        "dependency set"
                    ),
                    "notes": [
                        (
                            "the constraints output type does not support "
                            "extras"
                        ),
                        (
                            "separate the constraints and conda "
                            "dependency lists to preserve extras"
                        ),
                    ],
                },
            ],
            id="mixed-conda",
        ),
        pytest.param(
            """\
            + dependencies:
            +   test:
            +     common:
            +       - output_types:
            +           - requirements
            :             ~~~~~~~~~~~~warnings.0.notes.1
            +           - constraints
            :             ~~~~~~~~~~~warnings.0.notes.0
            +           - pyproject
            :             ~~~~~~~~~warnings.0.notes.2
            +         packages:
            +           - package[extra]
            :             ~~~~~~~~~~~~~~warnings.0.warning
            """,
            [
                {
                    "warning": (
                        'package "package" has extras in a constraints '
                        "dependency set"
                    ),
                    "notes": [
                        "the constraints output type does not support extras",
                        "separate the constraints and requirements "
                        "dependency lists to preserve extras",
                        "separate the constraints and pyproject "
                        "dependency lists to preserve extras",
                    ],
                },
            ],
            id="multiple-other-output-types",
        ),
        pytest.param(
            """\
            + dependencies:
            +   test:
            +     common:
            +       - packages:
            +           - 'package[foo,bar]>=1; python_version >= "3.12"'
            +           - --extra-index-url=https://example.com/[extra]
            +           - not a requirement
            +           - other>=2
            +         output_types: requirements
            """,
            [],
            id="requirements",
        ),
        pytest.param(
            """\
            + dependencies:
            +   test:
            +     common:
            +       - packages:
            +           - 'package[foo,bar]>=1; python_version >= "3.12"'
            +           - --extra-index-url=https://example.com/[extra]
            +           - not a requirement
            +           - other>=2
            +         output_types: pyproject
            """,
            [],
            id="pyproject",
        ),
        pytest.param(
            """\
            + dependencies:
            +   test:
            +     common:
            +       - packages:
            +           - 'package[foo,bar]>=1; python_version >= "3.12"'
            +           - --extra-index-url=https://example.com/[extra]
            +           - not a requirement
            +           - other>=2
            +         output_types: conda
            """,
            [],
            id="conda",
        ),
        pytest.param(
            """\
            + dependencies:
            +   test:
            +     common:
            +       - packages:
            +           - 'package[foo,bar]>=1; python_version >= "3.12"'
            +           - --extra-index-url=https://example.com/[extra]
            +           - not a requirement
            +           - other>=2
            +         output_types: []
            """,
            [],
            id="no-output-types",
        ),
        pytest.param(
            """\
            + dependencies:
            +   test:
            +     common:
            +       - output_types: requirements
            +         packages: ["package[extra]"]
            +     specific:
            +       - matrices:
            +           - matrix: {cuda: "12"}
            +             packages:
            +               - package[extra]==1
            :                 ~~~~~~~~~~~~~~~~~warnings.0.warning
            :                 ~~~~~~~~~~~~~~~~~warnings.0.replacements.0
            +           - matrix: {cuda: "13"}
            +             packages:
            +               - package[extra]==2
            :                 ~~~~~~~~~~~~~~~~~warnings.1.warning
            :                 ~~~~~~~~~~~~~~~~~warnings.1.replacements.0
            +         output_types: constraints
            :                       ~~~~~~~~~~~warnings.0.notes.0
            :                       ~~~~~~~~~~~warnings.1.notes.0
            +       - output_types: requirements
            +         matrices:
            +           - packages:
            +               - package[extra]
            """,  # noqa: E501
            [
                {
                    "warning": (
                        'package "package" has extras in a constraints '
                        "dependency set"
                    ),
                    "replacements": [
                        ("package==1"),
                    ],
                    "notes": [
                        (
                            "the constraints output type does not support "
                            "extras"
                        ),
                    ],
                },
                {
                    "warning": (
                        'package "package" has extras in a constraints '
                        "dependency set"
                    ),
                    "replacements": [
                        ("package==2"),
                    ],
                    "notes": [
                        (
                            "the constraints output type does not support "
                            "extras"
                        ),
                    ],
                },
            ],
            id="specific-and-common-contexts",
        ),
        pytest.param(
            """\
            + dependencies:
            +   test:
            +     common:
            +       - output_types: constraints
            :                       ~~~~~~~~~~~warnings.0.notes.0
            +         packages:
            +           - package[extra]
            :             ~~~~~~~~~~~~~~warnings.0.warning
            :             ~~~~~~~~~~~~~~warnings.0.replacements.0
            """,  # noqa: E501
            [
                {
                    "warning": (
                        'package "package" has extras in a constraints '
                        "dependency set"
                    ),
                    "replacements": [
                        ("package"),
                    ],
                    "notes": [
                        (
                            "the constraints output type does not support "
                            "extras"
                        ),
                    ],
                },
            ],
            id="bare-package",
        ),
        pytest.param(
            """\
            + dependencies:
            +   test:
            +     common:
            +       - output_types: constraints
            :                       ~~~~~~~~~~~warnings.0.notes.0
            +         packages:
            +           - 'package[extra]>=1'
            :             ~~~~~~~~~~~~~~~~~~~warnings.0.warning
            :             ~~~~~~~~~~~~~~~~~~~warnings.0.replacements.0
            """,  # noqa: E501
            [
                {
                    "warning": (
                        'package "package" has extras in a constraints '
                        "dependency set"
                    ),
                    "replacements": [
                        ("'package>=1'"),
                    ],
                    "notes": [
                        (
                            "the constraints output type does not support "
                            "extras"
                        ),
                    ],
                },
            ],
            id="quoted-package",
        ),
        pytest.param(
            """\
            + dependencies:
            +   test:
            +     common:
            +       - output_types: constraints
            :                       ~~~~~~~~~~~warnings.0.notes.0
            +         packages:
            +           - &pkg package[extra]==1
            :             ~~~~~~~~~~~~~~~~~~~~~~warnings.0.warning
            :             ~~~~~~~~~~~~~~~~~~~~~~warnings.0.replacements.0
            """,  # noqa: E501
            [
                {
                    "warning": (
                        'package "package" has extras in a constraints '
                        "dependency set"
                    ),
                    "replacements": [
                        ("&pkg package==1"),
                    ],
                    "notes": [
                        (
                            "the constraints output type does not support "
                            "extras"
                        ),
                    ],
                },
            ],
            id="anchor-definition",
        ),
        pytest.param(
            """\
            + dependencies:
            +   test:
            +     common:
            +       - output_types: requirements
            +         packages:
            +           - &pkg package[extra]==1
            :             ~~~~~~~~~~~~~~~~~~~~~~warnings.0.warning
            +       - output_types: constraints
            :                       ~~~~~~~~~~~warnings.0.notes.0
            +         packages:
            +           - *pkg
            """,  # noqa: E501
            [
                {
                    "warning": (
                        'package "package" has extras in a constraints '
                        "dependency set"
                    ),
                    "notes": [
                        (
                            "the constraints output type does not support "
                            "extras"
                        ),
                    ],
                },
            ],
            id="anchor-reference",
        ),
        pytest.param(
            """\
            + dependencies:
            +   test:
            +     common:
            +       - output_types: constraints
            :                       ~~~~~~~~~~~warnings.0.notes.0
            +         packages:
            +           - package[extra] @ https://example.com/package.whl
            :             ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~warnings.0.warning
            :             ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~warnings.0.replacements.0
            """,  # noqa: E501
            [
                {
                    "warning": (
                        'package "package" has extras in a constraints '
                        "dependency set"
                    ),
                    "replacements": [
                        ("package @ https://example.com/package.whl"),
                    ],
                    "notes": [
                        (
                            "the constraints output type does not support "
                            "extras"
                        ),
                    ],
                },
            ],
            id="direct-url",
        ),
    ],
)
def test_constraints_with_extras(content, warnings):
    content, spans = parse_named_spans(content, dict)
    linter = Linter("dependencies.yaml", content, "verify-dependencies")
    traverse_dependencies_yaml(ConstraintsWithExtrasHandler(linter), content)
    assert linter.warnings == zip_expected_warnings(
        spans.get("warnings", []), warnings
    )
