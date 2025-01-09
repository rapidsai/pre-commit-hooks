# Copyright (c) 2025, NVIDIA CORPORATION.
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
class RequiredCodeowners:
    file: str
    owners: list[CodeownersTransform]
    allow_extra: bool = False


def hard_coded_codeowners(owners: str) -> CodeownersTransform:
    return lambda *, project_prefix: owners


def cmake_codeowners(*, project_prefix: str) -> str:
    return f"@rapidsai/{project_prefix}-cmake-codeowners"


REQUIRED_CODEOWNERS = [
    RequiredCodeowners(
        file="CMakeLists.txt",
        owners=[
            cmake_codeowners,
        ],
    ),
    RequiredCodeowners(
        file="pyproject.toml",
        owners=[
            hard_coded_codeowners("@rapidsai/ci-codeowners"),
        ],
    ),
]


def parse_codeowners_line(line: str, skip: int) -> CodeownersLine | None:
    line_match = CODEOWNERS_LINE_RE.search(line)
    if not line_match:
        return None

    file_pattern = FilePattern(
        filename=line_match.group("file"),
        pos=(line_match.span("file")[0] + skip, line_match.span("file")[1] + skip),
    )
    owners: list[Owner] = []

    line_skip = skip + len(line_match.group("file"))
    for owner_match in CODEOWNERS_OWNER_RE.finditer(line_match.group("owners")):
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
    found_files: set[str],
) -> None:
    for required_codeowners in REQUIRED_CODEOWNERS:
        if required_codeowners.file == codeowners_line.file.filename:
            required_owners = [
                required_owner(project_prefix=args.project_prefix)
                for required_owner in required_codeowners.owners
            ]

            warning: LintWarning | None = None

            if not required_codeowners.allow_extra:
                extraneous_owners: list[Owner] = []
                for owner in codeowners_line.owners:
                    if owner.owner not in required_owners:
                        extraneous_owners.append(owner)
                if len(extraneous_owners) != 0:
                    if not warning:
                        warning = linter.add_warning(
                            codeowners_line.file.pos,
                            f"file '{codeowners_line.file.filename}' has incorrect "
                            "owners",
                        )
                    for owner in extraneous_owners:
                        warning.add_replacement(owner.pos_with_leading_whitespace, "")

            missing_required_owners: list[str] = []
            for required_owner in required_owners:
                for owner in codeowners_line.owners:
                    if required_owner == owner.owner:
                        break
                else:
                    missing_required_owners.append(required_owner)
            if len(missing_required_owners) != 0:
                if not warning:
                    warning = linter.add_warning(
                        codeowners_line.file.pos,
                        f"file '{codeowners_line.file.filename}' has incorrect owners",
                    )
                extra_string = ""
                for missing_required_owner in missing_required_owners:
                    extra_string += f" {missing_required_owner}"
                last = codeowners_line.owners[-1].pos[1]
                warning.add_replacement((last, last), extra_string)

            found_files.add(codeowners_line.file.filename)
            break


def check_codeowners(linter: Linter, args: argparse.Namespace) -> None:
    found_files: set[str] = set()
    for pos in linter.lines:
        line = linter.content[pos[0] : pos[1]]
        codeowners_line = parse_codeowners_line(line, pos[0])
        if codeowners_line:
            check_codeowners_line(linter, args, codeowners_line, found_files)

    new_text = ""
    for required_codeowners in REQUIRED_CODEOWNERS:
        if required_codeowners.file not in found_files:
            new_text += (
                f"{required_codeowners.file} "
                f"""{' '.join(owner(project_prefix=args.project_prefix)
                        for owner in required_codeowners.owners)}\n"""
            )
    if new_text:
        if len(linter.content) != 0 and linter.content[-1] != "\n":
            new_text = f"\n{new_text}"
        content_len = len(linter.content)
        linter.add_warning((0, 0), "missing required codeowners").add_replacement(
            (content_len, content_len), new_text
        )


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
    with m.execute() as ctx:
        ctx.add_check(check_codeowners)


if __name__ == "__main__":
    main()
