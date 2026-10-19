# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import bisect
import contextlib
import dataclasses
import functools
import re
import warnings
from collections.abc import Callable
from typing import ParamSpec, TYPE_CHECKING

from rich.console import Console
from rich.markup import escape

from . import agents

if TYPE_CHECKING:
    from collections.abc import Generator, Iterator
    from typing import Optional

Span = tuple[int, int]


Args = ParamSpec("Args")


def agent_instructions(
    func: "Callable[Args, str]",
) -> "Callable[Args, Callable[[], str]]":
    @functools.wraps(func)
    def instructions(
        *args: Args.args, **kwargs: Args.kwargs
    ) -> "Callable[[], str]":
        return lambda: func(*args, **kwargs)

    return instructions


class BinaryFileWarning(Warning):
    pass


@dataclasses.dataclass
class Replacement:
    span: Span
    newtext: str
    conflict: bool = False


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
    agent_instructions: "Optional[Callable[[], str]]" = dataclasses.field(
        default=None, compare=False
    )

    def add_replacement(self, span: Span, newtext: str) -> None:
        self.replacements.append(Replacement(span, newtext))

    def add_note(self, span: Span, msg: str) -> None:
        self.notes.append(Note(span, msg))

    def add_agent_instructions(
        self, instructions: "Callable[[], str]"
    ) -> None:
        self.agent_instructions = instructions


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
        self.sorted_replacements: "Optional[list[Replacement]]" = None
        self.has_suppressed_agent_instructions: bool = False

    def add_warning(self, span: Span, msg: str) -> LintWarning:
        w = LintWarning(span, msg)
        self.warnings.append(w)
        return w

    def fix(self) -> str:
        cursor = 0
        replaced_content = ""
        assert self.sorted_replacements is not None
        for replacement in self.sorted_replacements:
            if not replacement.conflict:
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
        verbose: bool = False,
    ) -> None:
        line_index = self.lines.line_for_pos(span[0])
        line_span = self.lines.spans[line_index]
        self.console.print(
            f"In file [bold]{escape(self.filename)}:{line_index + 1}:"
            f"{span[0] - line_span[0] + 1}[/bold]:"
        )
        self._print_highlighted_code(span, newtext, verbose)
        self.console.print(f"[bold]{note_type}:[/bold] {escape(msg)}")
        self.console.print()

    def print_warnings(self, args: "argparse.Namespace") -> None:
        sorted_warnings = sorted(
            self.get_enabled_warnings(),
            key=lambda warning: warning.span,
        )

        verbose = self.verbose_output(args)

        for warning in sorted_warnings:
            self._print_note("warning", warning.span, warning.msg)

            for note in warning.notes:
                self._print_note("note", note.span, note.msg)

            for replacement in warning.replacements:
                line_span_index = self.lines.line_for_pos(replacement.span[0])
                line_span = self.lines.spans[line_span_index]
                long = not verbose and (
                    self._NEWLINE_RE.search(replacement.newtext) is not None
                    or replacement.span[1] > line_span[1]
                )

                if args.fix:
                    if replacement.conflict:
                        if long:
                            replacement_msg = (
                                "suggested fix conflicts with another fix "
                                "and is too long to display"
                            )
                        else:
                            replacement_msg = (
                                "suggested fix conflicts with another fix"
                            )
                    else:
                        if long:
                            replacement_msg = (
                                "suggested fix applied but is too long to "
                                "display"
                            )
                        else:
                            replacement_msg = "suggested fix applied"
                else:
                    if long:
                        if replacement.conflict:
                            replacement_msg = (
                                "suggested fix conflicts with another fix "
                                "and is too long to display"
                            )
                        else:
                            replacement_msg = (
                                "suggested fix is too long to display, use "
                                "--fix to apply it"
                            )
                    else:
                        replacement_msg = "suggested fix"
                self._print_note(
                    "note",
                    replacement.span,
                    replacement_msg,
                    replacement.newtext,
                    verbose,
                )

            if warning.agent_instructions is not None:
                self.print_agent_instructions(args, warning.agent_instructions)

        self.print_suppressed_agent_instrutions()

    def _print_highlighted_code(
        self,
        span: Span,
        replacement: str | None = None,
        verbose: bool = False,
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
            end_line_span = self.lines.spans[self.lines.line_for_pos(span[1])]
            prefix = self.content[line_span[0] : left]
            suffix = self.content[span[1] : end_line_span[1]]

            def print_diff_lines(text: str, marker: str, color: str) -> None:
                lines = Lines(text).spans
                # A trailing newline does not add an empty diff line unless
                # there is surviving text on that line.
                if text and lines[-1][0] == len(text) and not suffix:
                    lines = lines[:-1]
                for i, (begin, end) in enumerate(lines):
                    if i and not verbose:
                        break
                    line_prefix = prefix if i == 0 else ""
                    line_suffix = suffix if i == len(lines) - 1 else ""
                    self.console.print(
                        f"[{color}]{marker}{escape(line_prefix)}"
                        f"[bold]{escape(text[begin:end])}[/bold]"
                        f"{escape(line_suffix)}[/{color}]"
                    )

            print_diff_lines(self.content[left : span[1]], "-", "red")
            print_diff_lines(replacement, "+", "green")

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

    @classmethod
    def verbose_output(
        cls, args: "argparse.Namespace", level: int = 1
    ) -> bool:
        return args.verbose >= level or agents.is_agent_environment()

    def get_enabled_warnings(self) -> "Iterator[LintWarning]":
        return filter(
            lambda w: Linter.is_warning_span_enabled(
                self.disabled_enabled_boundaries, w.span
            ),
            self.warnings,
        )

    def sort_and_check_replacements_for_conflicts(self) -> None:
        self.sorted_replacements = sorted(
            (
                replacement
                for warning in self.get_enabled_warnings()
                for replacement in warning.replacements
            ),
            key=lambda replacement: replacement.span,
        )

        for i, r1 in enumerate(self.sorted_replacements):
            for r2 in self.sorted_replacements[i + 1 :]:
                if r1.span[1] > r2.span[0] or r1.span == r2.span:
                    r1.conflict = True
                    r2.conflict = True
                else:
                    break

    @classmethod
    def format_agent_instructions(cls, instructions: str) -> list[str]:
        MAX_LINE_LENGTH = 79

        lines: list[str] = []
        for paragraph in filter(
            lambda p: p != "", re.split(r"\n\n+", instructions)
        ):
            if lines:
                lines.append("")
            if paragraph.startswith("  "):
                for line in paragraph.split("\n"):
                    assert line.startswith("  "), (
                        "Do not mix formatted and unformatted lines in a "
                        "paragraph"
                    )
                    lines.append(f"  {line}")
            else:
                assert "\n  " not in paragraph, (
                    "Do not mix formatted and unformatted lines in a paragraph"
                )
                line = ""
                sep = ""
                for word in filter(
                    lambda w: w != "", re.split(r"[ \n]", paragraph)
                ):
                    assert len(word) <= MAX_LINE_LENGTH, (
                        f'Word "{word}" is too long'
                    )
                    if len(f"  {line}{sep}{word}") > MAX_LINE_LENGTH:
                        lines.append(f"  {line}")
                        line = f"{word}"
                    else:
                        line += f"{sep}{word}"
                    sep = " "
                if line != "":
                    lines.append(f"  {line}")
        return lines

    def _print_agent_instructions(self, instructions: str) -> None:
        self.console.print("[bold]instructions for agents:[/bold]")
        for line in self.format_agent_instructions(instructions):
            self.console.print(escape(line))
        self.console.print()

    def print_agent_instructions(
        self, args: "argparse.Namespace", instructions: "Callable[[], str]"
    ) -> None:
        if self.verbose_output(args, 2):
            self._print_agent_instructions(instructions())
        else:
            self.has_suppressed_agent_instructions = True

    def print_suppressed_agent_instrutions(self) -> None:
        if self.has_suppressed_agent_instructions:
            self._print_agent_instructions(
                "Run this tool again with AI_AGENT=1 or -vv to see detailed "
                "instructions for agents."
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

            linter.sort_and_check_replacements_for_conflicts()
            linter.print_warnings(self.args)
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
            "--fix",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="whether or not to automatically fix warnings "
            "(yes by default)",
        )
        self.argparser.add_argument(
            "-v",
            "--verbose",
            action="count",
            help="show more verbose output (running with AI_AGENT=1 implies "
            "maximum verbosity)",
            default=0,
        )
        self.argparser.add_argument("files", nargs="+", metavar="file")

    def execute(self) -> ExecutionContext:
        return self.context_class(
            self.warning_name, self.argparser.parse_args()
        )
