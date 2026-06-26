# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock, patch

import pytest

from rapids_pre_commit_hooks import codeowners
from rapids_pre_commit_hooks.lint import Linter, LintWarning, Note, Replacement
from rapids_pre_commit_hooks_test_utils import parse_named_spans

MOCK_REQUIRED_CODEOWNERS_LINES = [
    codeowners.RequiredCodeownersLine(
        file="CMakeLists.txt",
        owners=[
            codeowners.project_codeowners("cmake"),
        ],
    ),
    codeowners.RequiredCodeownersLine(
        file="pyproject.toml",
        owners=[
            codeowners.hard_coded_codeowners("@rapidsai/ci-codeowners"),
        ],
        allow_extra=True,
        after=["CMakeLists.txt"],
    ),
]

patch_required_codeowners_lines = patch(
    "rapids_pre_commit_hooks.codeowners.required_codeowners_lines",
    lambda _args: MOCK_REQUIRED_CODEOWNERS_LINES,
)


@pytest.mark.parametrize(
    ["content", "skip"],
    [
        pytest.param(
            """\
            > filename @owner1 @owner2
            : ~~~~~~~~filename
            :          ~~~~~~~owners.0.span
            :         ~~~~~~~~owners.0.span_with_leading_whitespace
            :                  ~~~~~~~owners.1.span
            :                 ~~~~~~~~owners.1.span_with_leading_whitespace
            """,
            0,
            id="basic",
        ),
        pytest.param(
            """\
            > filename @owner1 @owner2
            : ~~~~~~~~filename
            :          ~~~~~~~owners.0.span
            :         ~~~~~~~~owners.0.span_with_leading_whitespace
            :                  ~~~~~~~owners.1.span
            :                 ~~~~~~~~owners.1.span_with_leading_whitespace
            """,
            1,
            id="skip",
        ),
        pytest.param(
            # Spans are deliberately misaligned because of the \t
            """\
            > filename\t @owner1  @owner2  # Comment
            : ~~~~~~~~filename
            :           ~~~~~~~owners.0.span
            :         ~~~~~~~~~owners.0.span_with_leading_whitespace
            :                    ~~~~~~~owners.1.span
            :                  ~~~~~~~~~owners.1.span_with_leading_whitespace
            """,
            0,
            id="whitespace-and-comment",
        ),
        pytest.param(
            # Spans are deliberately misaligned because of the escaped
            # backslashes
            """\
            > file\\ name @owner\\ 1 @owner\\ 2
            : ~~~~~~~~~~filename
            :            ~~~~~~~~~owners.0.span
            :           ~~~~~~~~~~owners.0.span_with_leading_whitespace
            :                      ~~~~~~~~~owners.1.span
            :                     ~~~~~~~~~~owners.1.span_with_leading_whitespace
            """,  # noqa: E501
            0,
            id="backslashes",
        ),
        pytest.param(
            "",
            0,
            id="empty",
        ),
        pytest.param(
            "> # comment",
            0,
            id="empty-with-comment",
        ),
    ],
)
def test_parse_codeowners_line(content, skip):
    content, spans = parse_named_spans(content, root_type=dict)
    try:
        filename = spans["filename"]
        owners = spans["owners"]
    except KeyError:
        codeowners_line = None
    else:
        codeowners_line = codeowners.CodeownersLine(
            file=codeowners.FilePattern(
                filename=content[slice(*filename)],
                span=tuple(c + skip for c in filename),
            ),
            owners=[
                codeowners.Owner(
                    owner=content[slice(*owner["span"])],
                    span=tuple(c + skip for c in owner["span"]),
                    span_with_leading_whitespace=tuple(
                        c + skip for c in owner["span_with_leading_whitespace"]
                    ),
                )
                for owner in owners
            ],
        )
    assert codeowners.parse_codeowners_line(content, skip) == codeowners_line


@pytest.mark.parametrize(
    ["content", "warnings"],
    [
        pytest.param(
            """\
            > CMakeLists.txt @rapidsai/cudf-cmake-codeowners
            : ~~~~~~~~~~~~~~filename
            """,
            [],
            id="good",
        ),
        pytest.param(
            """\
            > CMakeLists.txt @someone-else  # comment
            : ~~~~~~~~~~~~~~filename
            : ~~~~~~~~~~~~~~warnings.0.span
            :               ~~~~~~~~~~~~~~warnings.0.replacements.0
            :                             ^warnings.0.replacements.1
            """,
            [
                {
                    "replacements": [
                        "",
                        " @rapidsai/cudf-cmake-codeowners",
                    ],
                },
            ],
            id="wrong-owner",
        ),
        pytest.param(
            """\
            > CMakeLists.txt @someone-else @rapidsai/cudf-cmake-codeowners
            : ~~~~~~~~~~~~~~filename
            : ~~~~~~~~~~~~~~warnings.0.span
            :               ~~~~~~~~~~~~~~warnings.0.replacements.0
            """,
            [
                {
                    "replacements": [
                        "",
                    ],
                },
            ],
            id="extraneous-owner",
        ),
        pytest.param(
            """\
            > pyproject.toml @someone-else @rapidsai/ci-codeowners
            : ~~~~~~~~~~~~~~filename
            """,
            [],
            id="unchecked",
        ),
    ],
)
@patch_required_codeowners_lines
def test_check_codeowners_line(content, warnings):
    content, spans = parse_named_spans(content)
    warnings = [
        LintWarning(
            span=warning_spans["span"],
            msg=f"file '{content[slice(*spans['filename'])]}'"
            " has incorrect owners",
            replacements=[
                Replacement(span=replacement_span, newtext=replacement)
                for replacement, replacement_span in zip(
                    warning["replacements"],
                    warning_spans["replacements"],
                    strict=True,
                )
            ],
        )
        for warning, warning_spans in zip(
            warnings, spans.get("warnings", []), strict=True
        )
    ]

    codeowners_line = codeowners.parse_codeowners_line(content, 0)
    linter = Linter(".github/CODEOWNERS", content, "verify-codeowners")
    found_files = []
    codeowners.check_codeowners_line(
        linter, Mock(project_prefix="cudf"), codeowners_line, found_files
    )
    assert linter.warnings == warnings
    assert found_files == [
        (line, spans["filename"])
        for line in MOCK_REQUIRED_CODEOWNERS_LINES
        if line.file == codeowners_line.file.filename
    ]


@pytest.mark.parametrize(
    ["content", "warnings"],
    [
        pytest.param(
            """\
            +
            + CMakeLists.txt @rapidsai/cudf-cmake-codeowners
            + pyproject.toml @rapidsai/ci-codeowners
            """,
            [],
            id="good",
        ),
        pytest.param(
            """\
            +
            + CMakeLists.txt @someone-else
            : ~~~~~~~~~~~~~~0.span
            :               ~~~~~~~~~~~~~~0.replacements.0
            :                             ^0.replacements.1
            + pyproject.toml @rapidsai/ci-codeowners
            """,
            [
                {
                    "msg": "file 'CMakeLists.txt' has incorrect owners",
                    "notes": [],
                    "replacements": [
                        "",
                        " @rapidsai/cudf-cmake-codeowners",
                    ],
                },
            ],
            id="wrong-owners",
        ),
        pytest.param(
            """\
            +
            : ^0.span
            + pyproject.toml @rapidsai/ci-codeowners
            :                                        ^0.replacements.0
            """,
            [
                {
                    "msg": "missing required codeowners",
                    "notes": [],
                    "replacements": [
                        "CMakeLists.txt @rapidsai/cudf-cmake-codeowners\n",
                    ],
                },
            ],
            id="missing-files",
        ),
        pytest.param(
            """\
            +
            : ^0.span
            > pyproject.toml @rapidsai/ci-codeowners
            :                                       ^0.replacements.0
            """,
            [
                {
                    "msg": "missing required codeowners",
                    "notes": [],
                    "replacements": [
                        "\nCMakeLists.txt @rapidsai/cudf-cmake-codeowners\n",
                    ],
                },
            ],
            id="missing-files-no-newline",
        ),
        pytest.param(
            """\
            +
            + pyproject.toml @rapidsai/ci-codeowners
            : ~~~~~~~~~~~~~~0.span
            + CMakeLists.txt @rapidsai/cudf-cmake-codeowners
            : ~~~~~~~~~~~~~~0.notes.0
            """,
            [
                {
                    "msg": "file 'pyproject.toml' should come after "
                    "'CMakeLists.txt'",
                    "notes": [
                        "file 'CMakeLists.txt' is here",
                    ],
                    "replacements": [],
                },
            ],
            id="wrong-order",
        ),
    ],
)
@patch_required_codeowners_lines
def test_check_codeowners(content, warnings):
    content, spans = parse_named_spans(content, root_type=list)
    warnings = [
        LintWarning(
            span=warning_spans["span"],
            msg=warning["msg"],
            notes=[
                Note(
                    span=note_span,
                    msg=note,
                )
                for note, note_span in zip(
                    warning["notes"],
                    warning_spans.get("notes", []),
                    strict=True,
                )
            ],
            replacements=[
                Replacement(
                    span=replacement_span,
                    newtext=replacement,
                )
                for replacement, replacement_span in zip(
                    warning["replacements"],
                    warning_spans.get("replacements", []),
                    strict=True,
                )
            ],
        )
        for warning, warning_spans in zip(warnings, spans, strict=True)
    ]

    linter = Linter(".github/CODEOWNERS", content, "verify-codeowners")
    codeowners.check_codeowners(linter, Mock(project_prefix="cudf"))
    assert linter.warnings == warnings
