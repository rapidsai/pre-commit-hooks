# Copyright (c) 2024-2025, NVIDIA CORPORATION.
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
import bisect
import contextlib
import dataclasses
import functools
import re
import warnings
from collections.abc import Callable
from itertools import pairwise

from rich.console import Console
from rich.markup import escape

_PosType = tuple[int, int]


class OverlappingReplacementsError(RuntimeError):
    pass


class BinaryFileWarning(Warning):
    pass


@dataclasses.dataclass
class Replacement:
    pos: _PosType
    newtext: str


@dataclasses.dataclass
class Note:
    pos: _PosType
    msg: str


@dataclasses.dataclass
class LintWarning:
    pos: _PosType
    msg: str
    replacements: list[Replacement] = dataclasses.field(
        default_factory=list, kw_only=True
    )
    notes: list[Note] = dataclasses.field(default_factory=list, kw_only=True)

    def add_replacement(self, pos: _PosType, newtext: str) -> None:
        self.replacements.append(Replacement(pos, newtext))

    def add_note(self, pos: _PosType, msg: str) -> None:
        self.notes.append(Note(pos, msg))


class Lines:
    @functools.total_ordering
    class _LineComparator:
        def __init__(self, pos: _PosType) -> None:
            self.pos: _PosType = pos

        def __lt__(self, other: object) -> bool:
            assert isinstance(other, int)
            return self.pos[1] < other

        def __gt__(self, other: object) -> bool:
            assert isinstance(other, int)
            return self.pos[0] > other

        def __eq__(self, other: object) -> bool:
            assert isinstance(other, int)
            return self.pos[0] <= other <= self.pos[1]

    def __init__(self, content: str) -> None:
        self.content: str = content
        self.pos: list[_PosType] = []

        line_begin = 0
        line_end = 0
        state = "c"

        for c in content:
            if state == "c":
                if c == "\r":
                    self.pos.append((line_begin, line_end))
                    line_end = line_begin = line_end + 1
                    state = "r"
                elif c == "\n":
                    self.pos.append((line_begin, line_end))
                    line_end = line_begin = line_end + 1
                else:
                    line_end += 1
            elif state == "r":
                if c == "\r":
                    self.pos.append((line_begin, line_end))
                    line_end = line_begin = line_end + 1
                elif c == "\n":
                    line_end = line_begin = line_end + 1
                    state = "c"
                else:
                    line_end += 1
                    state = "c"

        self.pos.append((line_begin, line_end))

    def line_for_pos(self, index: int) -> int:
        line_index = bisect.bisect_left(
            [Lines._LineComparator(line) for line in self.pos], index
        )
        try:
            line_pos = self.pos[line_index]
        except IndexError:
            raise IndexError(f"Position {index} is not in the string")
        if not (line_pos[0] <= index <= line_pos[1]):
            raise IndexError(f"Position {index} is inside a line separator")
        return line_index


class Linter:
    NEWLINE_RE: re.Pattern = re.compile("[\r\n]")

    def __init__(self, filename: str, content: str) -> None:
        self.filename: str = filename
        self.content: str = content
        self.warnings: list[LintWarning] = []
        self.console: "Console" = Console(highlight=False)
        self.lines = Lines(content)

    def add_warning(self, pos: _PosType, msg: str) -> LintWarning:
        w = LintWarning(pos, msg)
        self.warnings.append(w)
        return w

    def fix(self) -> str:
        sorted_replacements = sorted(
            (
                replacement
                for warning in self.warnings
                for replacement in warning.replacements
            ),
            key=lambda replacement: replacement.pos,
        )

        for r1, r2 in pairwise(sorted_replacements):
            if r1.pos[1] > r2.pos[0]:
                raise OverlappingReplacementsError(f"{r1} overlaps with {r2}")

        cursor = 0
        replaced_content = ""
        for replacement in sorted_replacements:
            replaced_content += self.content[cursor : replacement.pos[0]]
            replaced_content += replacement.newtext
            cursor = replacement.pos[1]

        replaced_content += self.content[cursor:]
        return replaced_content

    def _print_note(
        self,
        note_type: str,
        pos: _PosType,
        msg: str,
        newtext: str | None = None,
    ) -> None:
        line_index = self.lines.line_for_pos(pos[0])
        line_pos = self.lines.pos[line_index]
        self.console.print(
            f"In file [bold]{escape(self.filename)}:{line_index + 1}:"
            f"{pos[0] - line_pos[0] + 1}[/bold]:"
        )
        self._print_highlighted_code(pos, newtext)
        self.console.print(f"[bold]{note_type}:[/bold] {escape(msg)}")
        self.console.print()

    def print_warnings(self, fix_applied: bool = False) -> None:
        sorted_warnings = sorted(
            self.warnings, key=lambda warning: warning.pos
        )

        for warning in sorted_warnings:
            self._print_note("warning", warning.pos, warning.msg)

            for note in warning.notes:
                self._print_note("note", note.pos, note.msg)

            for replacement in warning.replacements:
                line_index = self.lines.line_for_pos(replacement.pos[0])
                line_pos = self.lines.pos[line_index]
                newtext = replacement.newtext
                if match := self.NEWLINE_RE.search(newtext):
                    newtext = newtext[: match.start()]
                    long = True
                else:
                    long = False
                if replacement.pos[1] > line_pos[1]:
                    long = True

                if fix_applied:
                    if long:
                        replacement_msg = (
                            "suggested fix applied but is too long to display"
                        )
                    else:
                        replacement_msg = "suggested fix applied"
                else:
                    if long:
                        replacement_msg = (
                            "suggested fix is too long to display, use --fix "
                            "to apply it"
                        )
                    else:
                        replacement_msg = "suggested fix"
                self._print_note(
                    "note", replacement.pos, replacement_msg, newtext
                )

    def _print_highlighted_code(
        self, pos: _PosType, replacement: str | None = None
    ) -> None:
        line_index = self.lines.line_for_pos(pos[0])
        line_pos = self.lines.pos[line_index]
        left = pos[0]

        if self.lines.line_for_pos(pos[1]) == line_index:
            right = pos[1]
        else:
            right = line_pos[1]

        if replacement is None:
            self.console.print(
                f" {escape(self.content[line_pos[0] : left])}"
                f"[bold]{escape(self.content[left:right])}[/bold]"
                f"{escape(self.content[right : line_pos[1]])}"
            )
        else:
            self.console.print(
                f"[red]-{escape(self.content[line_pos[0] : left])}"
                f"[bold]{escape(self.content[left:right])}[/bold]"
                f"{escape(self.content[right : line_pos[1]])}[/red]"
            )
            self.console.print(
                f"[green]+{escape(self.content[line_pos[0] : left])}"
                f"[bold]{escape(replacement)}[/bold]"
                f"{escape(self.content[right : line_pos[1]])}[/green]"
            )


class ExecutionContext(contextlib.AbstractContextManager):
    def __init__(self, args: argparse.Namespace) -> None:
        self.args: argparse.Namespace = args
        self.checks: list[Callable[[Linter, argparse.Namespace], None]] = []

    def add_check(
        self, check: Callable[[Linter, argparse.Namespace], None]
    ) -> None:
        self.checks.append(check)

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if exc_type:
            return

        has_warnings = False

        for file in self.args.files:
            with open(file) as f:
                try:
                    content = f.read()
                except UnicodeDecodeError:
                    warnings.warn(
                        f"Refusing to run text linter on binary file {file}.",
                        BinaryFileWarning,
                    )
                    continue

            linter = Linter(file, content)
            for check in self.checks:
                check(linter, self.args)

            linter.print_warnings(self.args.fix)
            if self.args.fix:
                fix = linter.fix()
                if fix != content:
                    with open(file, "w") as f:
                        f.write(fix)

            if len(linter.warnings) > 0:
                has_warnings = True

        if has_warnings:
            exit(1)


class LintMain:
    context_class = ExecutionContext

    def __init__(self) -> None:
        self.argparser: argparse.ArgumentParser = argparse.ArgumentParser()
        self.argparser.add_argument(
            "--fix", action="store_true", help="automatically fix warnings"
        )
        self.argparser.add_argument("files", nargs="+", metavar="file")

    def execute(self) -> ExecutionContext:
        return self.context_class(self.argparser.parse_args())
