# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import argparse
import dataclasses
import re
from typing import Protocol

from rapids_pre_commit_hooks.lint import Linter, LintMain, LintWarning

CODEOWNERS_OWNER_RE_STR = r"([^\n#\s\\]|\\[^\n])+"
CODEOWNERS_OWNER_RE = re.compile(rf"\s+(?P<owner>{CODEOWNERS_OWNER_RE_STR})")
CODEOWNERS_LINE_RE = re.compile(
    rf"^(?P<file>([^\n#\s\\]|\\[^\n])+)(?P<owners>(\s+{CODEOWNERS_OWNER_RE_STR})+)"
)


@dataclasses.dataclass
class FilePattern:
    filename: str
    pos: tuple[int, int]


@dataclasses.dataclass
class Owner:
    owner: str
    pos: tuple[int, int]
    pos_with_leading_whitespace: tuple[int, int]


@dataclasses.dataclass
class CodeownersLine:
    file: FilePattern
    owners: list[Owner]


class CodeownersTransform(Protocol):
    def __call__(self, *, project_prefix: str) -> str: ...


@dataclasses.dataclass
class RequiredCodeownersLine:
    file: str
    owners: list[CodeownersTransform]
    allow_extra: bool = False
    after: list[str] = dataclasses.field(default_factory=list)


def hard_coded_codeowners(owners: str) -> CodeownersTransform:
    return lambda *, project_prefix: owners  # noqa: ARG005


def project_codeowners(category: str) -> CodeownersTransform:
    return (
        lambda *,
        project_prefix: f"@rapidsai/{project_prefix}-{category}-codeowners"
    )


def required_codeowners_list(
    files: list[str], owners: list[CodeownersTransform], after: list[str] = []
) -> list[RequiredCodeownersLine]:
    return [
        RequiredCodeownersLine(file=file, owners=owners, after=after)
        for file in files
    ]


REQUIRED_CI_CODEOWNERS_LINES = required_codeowners_list(
    [
        "/.github/",
        "/ci/",
    ],
    [hard_coded_codeowners("@rapidsai/ci-codeowners")],
)
REQUIRED_PACKAGING_CODEOWNERS_LINES = required_codeowners_list(
    [
        "/conda/",
        "dependencies.yaml",
        "/build.sh",
        "pyproject.toml",
        "/.pre-commit-config.yaml",
        "/.devcontainer/",
    ],
    [hard_coded_codeowners("@rapidsai/packaging-codeowners")],
)
REQUIRED_CPP_CODEOWNERS_LINES = required_codeowners_list(
    [
        "cpp/",
    ],
    [project_codeowners("cpp")],
)
REQUIRED_PYTHON_CODEOWNERS_LINES = required_codeowners_list(
    [
        "python/",
    ],
    [project_codeowners("python")],
)
REQUIRED_CMAKE_CODEOWNERS_LINES = required_codeowners_list(
    [
        "CMakeLists.txt",
        "**/cmake/",
        "*.cmake",
    ],
    [project_codeowners("cmake")],
    [
        *(
            after
            for lines in [
                REQUIRED_CPP_CODEOWNERS_LINES,
                REQUIRED_PYTHON_CODEOWNERS_LINES,
            ]
            for line in lines
            for after in line.after
        ),
    ],
)


def required_codeowners_lines(
    args: argparse.Namespace,
) -> list[RequiredCodeownersLine]:
    return [
        *(REQUIRED_CI_CODEOWNERS_LINES if args.ci else []),
        *(REQUIRED_PACKAGING_CODEOWNERS_LINES if args.packaging else []),
        *(REQUIRED_CPP_CODEOWNERS_LINES if args.cpp else []),
        *(REQUIRED_PYTHON_CODEOWNERS_LINES if args.python else []),
        *(REQUIRED_CMAKE_CODEOWNERS_LINES if args.cmake else []),
    ]


def parse_codeowners_line(line: str, skip: int) -> CodeownersLine | None:
    line_match = CODEOWNERS_LINE_RE.search(line)
    if not line_match:
        return None

    file_pattern = FilePattern(
        filename=line_match.group("file"),
        pos=(
            line_match.span("file")[0] + skip,
            line_match.span("file")[1] + skip,
        ),
    )
    owners: list[Owner] = []

    line_skip = skip + len(line_match.group("file"))
    for owner_match in CODEOWNERS_OWNER_RE.finditer(
        line_match.group("owners")
    ):
        start, end = owner_match.span("owner")
        whitespace_start, _ = owner_match.span()
        owners.append(
            Owner(
                owner=owner_match.group("owner"),
                pos=(start + line_skip, end + line_skip),
                pos_with_leading_whitespace=(
                    whitespace_start + line_skip,
                    end + line_skip,
                ),
            )
        )

    return CodeownersLine(file=file_pattern, owners=owners)


def check_codeowners_line(
    linter: Linter,
    args: argparse.Namespace,
    codeowners_line: CodeownersLine,
    found_files: list[tuple[RequiredCodeownersLine, tuple[int, int]]],
) -> None:
    for required_codeowners_line in required_codeowners_lines(args):
        if required_codeowners_line.file == codeowners_line.file.filename:
            required_owners = [
                required_owner(project_prefix=args.project_prefix)
                for required_owner in required_codeowners_line.owners
            ]

            warning: LintWarning | None = None

            if not required_codeowners_line.allow_extra:
                extraneous_owners: list[Owner] = [
                    owner
                    for owner in codeowners_line.owners
                    if owner.owner not in required_owners
                ]
                if extraneous_owners:
                    warning = linter.add_warning(
                        codeowners_line.file.pos,
                        f"file '{codeowners_line.file.filename}' has "
                        "incorrect owners",
                    )
                    for owner in extraneous_owners:
                        warning.add_replacement(
                            owner.pos_with_leading_whitespace, ""
                        )

            missing_required_owners: list[str] = []
            for required_owner in required_owners:
                for owner in codeowners_line.owners:
                    if required_owner == owner.owner:
                        break
                else:
                    missing_required_owners.append(required_owner)
            if missing_required_owners:
                if not warning:
                    warning = linter.add_warning(
                        codeowners_line.file.pos,
                        f"file '{codeowners_line.file.filename}' has "
                        "incorrect owners",
                    )
                extra_string = " " + " ".join(missing_required_owners)
                last = codeowners_line.owners[-1].pos[1]
                warning.add_replacement((last, last), extra_string)

            for found_file, found_pos in found_files:
                if codeowners_line.file.filename in found_file.after:
                    linter.add_warning(
                        found_pos,
                        f"file '{found_file.file}' should come after "
                        f"'{codeowners_line.file.filename}'",
                    ).add_note(
                        codeowners_line.file.pos,
                        f"file '{codeowners_line.file.filename}' is here",
                    )

            found_files.append(
                (required_codeowners_line, codeowners_line.file.pos)
            )
            break


def check_codeowners(linter: Linter, args: argparse.Namespace) -> None:
    found_files: list[tuple[RequiredCodeownersLine, tuple[int, int]]] = []
    for begin, end in linter.lines.pos:
        line = linter.content[begin:end]
        codeowners_line = parse_codeowners_line(line, begin)
        if codeowners_line:
            check_codeowners_line(linter, args, codeowners_line, found_files)

    new_text = ""
    for required_codeowners_line in required_codeowners_lines(args):
        if required_codeowners_line.file not in map(
            lambda line: line[0].file, found_files
        ):
            owners_text = " ".join(
                owner(project_prefix=args.project_prefix)
                for owner in required_codeowners_line.owners
            )
            new_text += (
                f"{required_codeowners_line.file} {owners_text}"
                + linter.lines.newline_style
            )
    if new_text:
        if linter.content and not linter.content.endswith("\n"):
            new_text = f"{linter.lines.newline_style}{new_text}"
        content_len = len(linter.content)
        linter.add_warning(
            (0, 0), "missing required codeowners"
        ).add_replacement((content_len, content_len), new_text)


def main() -> None:
    m = LintMain()
    m.argparser.description = (
        "Verify that the CODEOWNERS file has the correct codeowners."
    )
    m.argparser.add_argument(
        "--project-prefix",
        metavar="<project prefix>",
        help="project prefix to insert for project-specific team names",
        required=True,
    )
    m.argparser.add_argument(
        "--ci",
        help="enforce rules for CI codeowners",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    m.argparser.add_argument(
        "--packaging",
        help="enforce rules for packaging codeowners",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    m.argparser.add_argument(
        "--cpp",
        help="enforce rules for C++ codeowners",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    m.argparser.add_argument(
        "--python",
        help="enforce rules for Python codeowners",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    m.argparser.add_argument(
        "--cmake",
        help="enforce rules for CMake codeowners",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    with m.execute() as ctx:
        ctx.add_check(check_codeowners)


if __name__ == "__main__":
    main()
