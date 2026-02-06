# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import argparse
import bisect
import contextlib
import dataclasses
import functools
import re
import warnings
from itertools import pairwise
from typing import TYPE_CHECKING

from rich.console import Console
from rich.markup import escape

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Iterator

Span = tuple[int, int]


class OverlappingReplacementsError(RuntimeError):
    pass


class BinaryFileWarning(Warning):
    pass


@dataclasses.dataclass
class Replacement:
    span: Span
    newtext: str


@dataclasses.dataclass
class Note:
    span: Span
    msg: str


@dataclasses.dataclass
class LintWarning:
    span: Span
    msg: str
    replacements: list[Replacement] = dataclasses.field(
        default_factory=list, kw_only=True
    )
    notes: list[Note] = dataclasses.field(default_factory=list, kw_only=True)

    def add_replacement(self, span: Span, newtext: str) -> None:
        self.replacements.append(Replacement(span, newtext))

    def add_note(self, span: Span, msg: str) -> None:
        self.notes.append(Note(span, msg))


class Lines:
    @functools.total_ordering
    class _LineComparator:
        def __init__(self, span: Span) -> None:
            self.span: Span = span

        def __lt__(self, other: object) -> bool:
            assert isinstance(other, int)
            return self.span[1] < other

        def __gt__(self, other: object) -> bool:
            assert isinstance(other, int)
            return self.span[0] > other

        def __eq__(self, other: object) -> bool:
            assert isinstance(other, int)
            return self.span[0] <= other <= self.span[1]

    def __init__(self, content: str) -> None:
        self.content: str = content
        self.spans: list[Span] = []

        line_begin = 0
        line_end = 0
        state = "c"

        self.newline_count = {
            "\n": 0,
            "\r\n": 0,
            "\r": 0,
        }

        for c in content:
            if state == "c":
                if c == "\r":
                    self.spans.append((line_begin, line_end))
                    line_end = line_begin = line_end + 1
                    state = "r"
                elif c == "\n":
                    self.spans.append((line_begin, line_end))
                    line_end = line_begin = line_end + 1
                    self.newline_count["\n"] += 1
                else:
                    line_end += 1
            elif state == "r":
                if c == "\r":
                    self.spans.append((line_begin, line_end))
                    line_end = line_begin = line_end + 1
                    self.newline_count["\r"] += 1
                elif c == "\n":
                    line_end = line_begin = line_end + 1
                    state = "c"
                    self.newline_count["\r\n"] += 1
                else:
                    line_end += 1
                    state = "c"
                    self.newline_count["\r"] += 1

        self.spans.append((line_begin, line_end))
        if state == "r":
            self.newline_count["\r"] += 1
        self.newline_style, _ = max(
            self.newline_count.items(),
            key=lambda item: item[1],
        )

    def line_for_pos(self, pos: int) -> int:
        line_span_index = bisect.bisect_left(
            [Lines._LineComparator(line) for line in self.spans], pos
        )
        try:
            line_span = self.spans[line_span_index]
        except IndexError:
            raise IndexError(f"Position {pos} is not in the string")
        if not (line_span[0] <= pos <= line_span[1]):
            raise IndexError(f"Position {pos} is inside a line separator")
        return line_span_index


class Linter:
    _NEWLINE_RE: re.Pattern = re.compile(r"[\r\n]")
    _DISABLE_ENABLE_DIRECTIVE_RE: re.Pattern = re.compile(
        r"\brapids-pre-commit-hooks: *"
        r"(?P<directive_name>enable|disable)"
        r"(?:-(?P<scope>next-line))?(?P<word_boundary>\b)?"
        r"(?: *\[(?P<warning_names>[\w-]+(?:,[\w-]+)*)\])?"
    )

    def __init__(self, filename: str, content: str, warning_name: str) -> None:
        self.filename: str = filename
        self.content: str = content
        self.warning_name: str = warning_name
        self.warnings: list[LintWarning] = []
        self.console: "Console" = Console(highlight=False)
        self.lines: "Lines" = Lines(content)
        self.disabled_enabled_boundaries = (
            Linter.get_disabled_enabled_boundaries(self.lines, warning_name)
        )

    def add_warning(self, span: Span, msg: str) -> LintWarning:
        w = LintWarning(span, msg)
        self.warnings.append(w)
        return w

    def fix(self) -> str:
        sorted_replacements = sorted(
            (
                replacement
                for warning in self.get_enabled_warnings()
                for replacement in warning.replacements
            ),
            key=lambda replacement: replacement.span,
        )

        for r1, r2 in pairwise(sorted_replacements):
            if r1.span[1] > r2.span[0]:
                raise OverlappingReplacementsError(f"{r1} overlaps with {r2}")

        cursor = 0
        replaced_content = ""
        for replacement in sorted_replacements:
            replaced_content += self.content[cursor : replacement.span[0]]
            replaced_content += replacement.newtext
            cursor = replacement.span[1]

        replaced_content += self.content[cursor:]
        return replaced_content

    def _print_note(
        self,
        note_type: str,
        span: Span,
        msg: str,
        newtext: str | None = None,
    ) -> None:
        line_index = self.lines.line_for_pos(span[0])
        line_span = self.lines.spans[line_index]
        self.console.print(
            f"In file [bold]{escape(self.filename)}:{line_index + 1}:"
            f"{span[0] - line_span[0] + 1}[/bold]:"
        )
        self._print_highlighted_code(span, newtext)
        self.console.print(f"[bold]{note_type}:[/bold] {escape(msg)}")
        self.console.print()

    def print_warnings(self, fix_applied: bool = False) -> None:
        sorted_warnings = sorted(
            self.get_enabled_warnings(),
            key=lambda warning: warning.span,
        )

        for warning in sorted_warnings:
            self._print_note("warning", warning.span, warning.msg)

            for note in warning.notes:
                self._print_note("note", note.span, note.msg)

            for replacement in warning.replacements:
                line_span_index = self.lines.line_for_pos(replacement.span[0])
                line_span = self.lines.spans[line_span_index]
                newtext = replacement.newtext
                if match := self._NEWLINE_RE.search(newtext):
                    newtext = newtext[: match.start()]
                    long = True
                else:
                    long = False
                if replacement.span[1] > line_span[1]:
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
                    "note", replacement.span, replacement_msg, newtext
                )

    def _print_highlighted_code(
        self, span: Span, replacement: str | None = None
    ) -> None:
        line_span_index = self.lines.line_for_pos(span[0])
        line_span = self.lines.spans[line_span_index]
        left = span[0]

        if self.lines.line_for_pos(span[1]) == line_span_index:
            right = span[1]
        else:
            right = line_span[1]

        if replacement is None:
            self.console.print(
                f" {escape(self.content[line_span[0] : left])}"
                f"[bold]{escape(self.content[left:right])}[/bold]"
                f"{escape(self.content[right : line_span[1]])}"
            )
        else:
            self.console.print(
                f"[red]-{escape(self.content[line_span[0] : left])}"
                f"[bold]{escape(self.content[left:right])}[/bold]"
                f"{escape(self.content[right : line_span[1]])}[/red]"
            )
            self.console.print(
                f"[green]+{escape(self.content[line_span[0] : left])}"
                f"[bold]{escape(replacement)}[/bold]"
                f"{escape(self.content[right : line_span[1]])}[/green]"
            )

    @classmethod
    def get_disabled_enabled_boundaries(
        cls,
        lines: Lines,
        warning_name: str,
    ) -> "list[tuple[Span, bool]]":
        def helper() -> "Generator[tuple[Span, bool]]":
            start = 0
            enabled = True
            next_line_directives: list[tuple[Span, bool]] = []

            def handle_end(end: int) -> "Generator[tuple[Span, bool]]":
                nonlocal start
                while True:
                    try:
                        next_line_span, next_line_enabled = (
                            next_line_directives.pop(0)
                        )
                    except IndexError:
                        break
                    if next_line_span[0] >= end:
                        next_line_directives.insert(
                            0, (next_line_span, next_line_enabled)
                        )
                        break
                    if next_line_span[0] >= start:
                        yield ((start, next_line_span[0]), enabled)
                    yield (next_line_span, next_line_enabled)
                    start = next_line_span[1]
                if start <= end:
                    yield ((start, end), enabled)

            for m in Linter._DISABLE_ENABLE_DIRECTIVE_RE.finditer(
                lines.content
            ):
                if m.group("word_boundary") is None:
                    continue
                if m.group("warning_names") and warning_name not in m.group(
                    "warning_names"
                ).split(","):
                    continue

                directive_is_enable = m.group("directive_name") == "enable"
                if m.group("scope") == "next-line":
                    this_line = lines.line_for_pos(m.start())
                    next_line = this_line + 1
                    if (
                        next_line_directives
                        and next_line_directives[-1][0]
                        == lines.spans[next_line]
                    ):
                        next_line_directives[-1] = (
                            lines.spans[next_line],
                            directive_is_enable,
                        )
                    else:
                        next_line_directives.append(
                            (lines.spans[next_line], directive_is_enable)
                        )
                else:
                    yield from handle_end(m.start())
                    start = max(start, m.start())
                    enabled = directive_is_enable

            yield from handle_end(len(lines.content))

        return list(helper())

    @classmethod
    def is_warning_span_enabled(
        cls, boundaries: list[tuple[Span, bool]], warning_span: Span
    ) -> bool:
        start = bisect.bisect_left(
            boundaries,
            warning_span[0],
            key=lambda b: b[0][0],
        )
        end = bisect.bisect_right(
            boundaries,
            warning_span[1],
            key=lambda b: b[0][1],
        )
        if warning_span[0] == warning_span[1]:
            move_start = (
                start > 0 and warning_span[0] <= boundaries[start - 1][0][1]
            )
            move_end = (
                end < len(boundaries)
                and warning_span[1] >= boundaries[end][0][0]
            )
        else:
            move_start = (
                start > 0 and warning_span[0] < boundaries[start - 1][0][1]
            )
            move_end = (
                end < len(boundaries)
                and warning_span[1] > boundaries[end][0][0]
            )
        if move_start:
            start -= 1
        if move_end:
            end += 1
        return any(map(lambda b: b[1], boundaries[start:end]))

    def get_enabled_warnings(self) -> "Iterator[LintWarning]":
        return filter(
            lambda w: Linter.is_warning_span_enabled(
                self.disabled_enabled_boundaries, w.span
            ),
            self.warnings,
        )


class ExecutionContext(contextlib.AbstractContextManager):
    def __init__(self, warning_name: str, args: argparse.Namespace) -> None:
        self.warning_name: str = warning_name
        self.args: argparse.Namespace = args
        self.checks: "list[Callable[[Linter, argparse.Namespace], None]]" = []

    def add_check(
        self, check: "Callable[[Linter, argparse.Namespace], None]"
    ) -> None:
        self.checks.append(check)

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if exc_type:
            return

        has_warnings = False

        for file in self.args.files:
            with open(file, newline="") as f:
                try:
                    content = f.read()
                except UnicodeDecodeError:
                    warnings.warn(
                        f"Refusing to run text linter on binary file {file}.",
                        BinaryFileWarning,
                    )
                    continue

            linter = Linter(file, content, self.warning_name)
            for check in self.checks:
                check(linter, self.args)

            linter.print_warnings(self.args.fix)
            if self.args.fix:
                fix = linter.fix()
                if fix != content:
                    with open(file, "w") as f:
                        f.write(fix)

            if any(linter.get_enabled_warnings()):
                has_warnings = True

        if has_warnings:
            exit(1)


class LintMain:
    context_class = ExecutionContext

    def __init__(self, warning_name: str) -> None:
        self.warning_name: str = warning_name
        self.argparser: argparse.ArgumentParser = argparse.ArgumentParser()
        self.argparser.add_argument(
            "--fix", action="store_true", help="automatically fix warnings"
        )
        self.argparser.add_argument("files", nargs="+", metavar="file")

    def execute(self) -> ExecutionContext:
        return self.context_class(
            self.warning_name, self.argparser.parse_args()
        )
