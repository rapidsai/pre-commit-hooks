# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import contextlib
import os.path
from textwrap import dedent
from unittest.mock import Mock, call, patch

import pytest

from rapids_pre_commit_hooks.lint import (
    BinaryFileWarning,
    Lines,
    Linter,
    LintMain,
    OverlappingReplacementsError,
)
from rapids_pre_commit_hooks_test_utils import parse_named_spans


class TestLines:
    LONG_CONTENTS = (
        "line 1\nline 2\rline 3\r\nline 4\r\n\nline 6\r\n\r\nline 8\n\r\n"
        "line 10\r\r\nline 12\r\n\rline 14\n\nline 16\r\rline 18\n\rline 20"
    )

    @pytest.mark.parametrize(
        [
            "content",
            "expected_spans",
            "expected_lf_count",
            "expected_crlf_count",
            "expected_cr_count",
            "expected_newline_style",
        ],
        [
            pytest.param(
                "line 1\n",
                [
                    (0, 6),
                    (7, 7),
                ],
                1,
                0,
                0,
                "\n",
                id="lf",
            ),
            pytest.param(
                "line 1\r\n",
                [
                    (0, 6),
                    (8, 8),
                ],
                0,
                1,
                0,
                "\r\n",
                id="crlf",
            ),
            pytest.param(
                "line 1\r",
                [
                    (0, 6),
                    (7, 7),
                ],
                0,
                0,
                1,
                "\r",
                id="cr",
            ),
            pytest.param(
                "",
                [
                    (0, 0),
                ],
                0,
                0,
                0,
                "\n",
                id="empty",
            ),
            pytest.param(
                LONG_CONTENTS,
                [
                    (0, 6),
                    (7, 13),
                    (14, 20),
                    (22, 28),
                    (30, 30),
                    (31, 37),
                    (39, 39),
                    (41, 47),
                    (48, 48),
                    (50, 57),
                    (58, 58),
                    (60, 67),
                    (69, 69),
                    (70, 77),
                    (78, 78),
                    (79, 86),
                    (87, 87),
                    (88, 95),
                    (96, 96),
                    (97, 104),
                ],
                6,
                7,
                6,
                "\r\n",
                id="complex",
            ),
            pytest.param(
                "a\nb\nc\r\nd\r\ne",
                [
                    (0, 1),
                    (2, 3),
                    (4, 5),
                    (7, 8),
                    (10, 11),
                ],
                2,
                2,
                0,
                "\n",
                id="tied",
            ),
        ],
    )
    def test_spans(
        self,
        content,
        expected_spans,
        expected_lf_count,
        expected_crlf_count,
        expected_cr_count,
        expected_newline_style,
    ):
        lines = Lines(content)
        assert lines.spans == expected_spans
        assert lines.newline_count == {
            "\n": expected_lf_count,
            "\r\n": expected_crlf_count,
            "\r": expected_cr_count,
        }
        assert lines.newline_style == expected_newline_style

    @pytest.mark.parametrize(
        ["contents", "pos", "line", "raises"],
        [
            (LONG_CONTENTS, 0, 0, contextlib.nullcontext()),
            (LONG_CONTENTS, 3, 0, contextlib.nullcontext()),
            (LONG_CONTENTS, 6, 0, contextlib.nullcontext()),
            (LONG_CONTENTS, 10, 1, contextlib.nullcontext()),
            (
                LONG_CONTENTS,
                21,
                None,
                pytest.raises(
                    IndexError,
                    match="^Position 21 is inside a line separator$",
                ),
            ),
            (LONG_CONTENTS, 34, 5, contextlib.nullcontext()),
            (LONG_CONTENTS, 97, 19, contextlib.nullcontext()),
            (LONG_CONTENTS, 104, 19, contextlib.nullcontext()),
            (
                LONG_CONTENTS,
                200,
                None,
                pytest.raises(
                    IndexError, match="^Position 200 is not in the string$"
                ),
            ),
            ("line 1", 0, 0, contextlib.nullcontext()),
            ("line 1", 3, 0, contextlib.nullcontext()),
            ("line 1", 6, 0, contextlib.nullcontext()),
        ],
    )
    def test_line_for_pos(
        self,
        contents,
        pos,
        line,
        raises,
    ):
        lines = Lines(contents)
        with raises:
            assert lines.line_for_pos(pos) == line


class TestLinter:
    def test_fix(self):
        content, spans = parse_named_spans(
            """\
            > Hello world!
            : ^no_fix
            :      ^punctuation
            : ~~~~~hello
            :            ~shout
            :       ~~~~~world
            :            ^noop
            """
        )
        linter = Linter("test.txt", content, "test")
        assert linter.fix() == "Hello world!"

        linter.add_warning(spans["no_fix"], "no fix")
        assert linter.fix() == "Hello world!"

        linter.add_warning(
            spans["punctuation"], "use punctuation"
        ).add_replacement(spans["punctuation"], ",")
        linter.add_warning(
            spans["hello"], "say good bye instead"
        ).add_replacement(spans["hello"], "Good bye")
        linter.add_warning(spans["shout"], "don't shout").add_replacement(
            spans["shout"], ""
        )
        linter.add_warning(
            spans["world"], "no-op replacement"
        ).add_replacement(spans["noop"], "")
        assert linter.fix() == "Good bye, world"

        linter.add_warning(spans["shout"], "don't shout").add_replacement(
            spans["shout"], "."
        )
        with pytest.raises(
            OverlappingReplacementsError,
            match=r"^Replacement\(span="
            rf"\({spans['shout'][0]}, {spans['shout'][1]}\),"
            r" newtext=''\) overlaps with "
            rf"Replacement\(span=\({spans['shout'][0]}, {spans['shout'][1]}\),"
            r" newtext='\.'\)$",
        ):
            linter.fix()

    def test_fix_disabled(self):
        content, spans = parse_named_spans(
            """\
            + # rapids-pre-commit-hooks: disable
            + Hello world!
            :            ~shout
            """
        )
        linter = Linter("test.txt", content, "test")
        linter.add_warning(spans["shout"], "don't shout").add_replacement(
            spans["shout"], ""
        )
        assert linter.fix() == content

    @pytest.mark.parametrize(
        ["content", "warning_name", "expected_boundaries"],
        [
            pytest.param(
                ": ^0",
                "test",
                [True],
                id="empty",
            ),
            pytest.param(
                """\
                + Hello
                : ~~~~~~0
                + world!
                : ~~~~~~~0
                """,
                "test",
                [True],
                id="content-with-no-directives",
            ),
            pytest.param(
                """\
                + # rapids-pre-commit-hooks:disable
                : ~~0
                :   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~1
                + Hello
                : ~~~~~~1
                """,
                "test",
                [True, False],
                id="single-unfiltered-disable",
            ),
            pytest.param(
                """\
                + # rapids-pre-commit-hooks: enable
                : ~~0
                :   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~1
                + Hello
                : ~~~~~~1
                """,
                "test",
                [True, True],
                id="single-unfiltered-enable",
            ),
            pytest.param(
                """\
                + # rapids-pre-commit-hooks: disable[relevant]
                : ~~0
                :   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~1
                + Hello
                : ~~~~~~1
                """,
                "relevant",
                [True, False],
                id="single-relevant-disable",
            ),
            pytest.param(
                """\
                + # rapids-pre-commit-hooks:enable [relevant]
                : ~~0
                :   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~1
                + Hello
                : ~~~~~~1
                """,
                "relevant",
                [True, True],
                id="single-relevant-enable",
            ),
            pytest.param(
                """\
                + # rapids-pre-commit-hooks:disable [irrelevant]
                : ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~0
                + Hello
                : ~~~~~~0
                """,
                "relevant",
                [True],
                id="single-irrelevant-disable",
            ),
            pytest.param(
                """\
                + # rapids-pre-commit-hooks: enable[irrelevant]
                : ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~0
                + Hello
                : ~~~~~~0
                """,
                "relevant",
                [True],
                id="single-irrelevant-enable",
            ),
            pytest.param(
                """\
                + # rapids-pre-commit-hooks: disable-next-line
                : ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~0
                + Hello
                : ~~~~~1
                :      ~2
                """,
                "test",
                [True, False, True],
                id="single-unfiltered-disable-next-line",
            ),
            pytest.param(
                """\
                + # rapids-pre-commit-hooks: enable-next-line
                : ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~0
                + Hello
                : ~~~~~1
                :      ~2
                """,
                "test",
                [True, True, True],
                id="single-unfiltered-enable-next-line",
            ),
            pytest.param(
                """\
                + # rapids-pre-commit-hooks: disable-next-line[relevant]
                : ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~0
                + Hello
                : ~~~~~1
                :      ~2
                """,
                "relevant",
                [True, False, True],
                id="single-relevant-disable-next-line",
            ),
            pytest.param(
                """\
                + # rapids-pre-commit-hooks: enable-next-line[relevant]
                : ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~0
                + Hello
                : ~~~~~1
                :      ~2
                """,
                "relevant",
                [True, True, True],
                id="single-relevant-enable-next-line",
            ),
            pytest.param(
                """\
                + # rapids-pre-commit-hooks: disable-next-line[irrelevant]
                : ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~0
                + Hello
                : ~~~~~~0
                """,
                "relevant",
                [True],
                id="single-irrelevant-disable-next-line",
            ),
            pytest.param(
                """\
                + # rapids-pre-commit-hooks: enable-next-line[irrelevant]
                : ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~0
                + Hello
                : ~~~~~~0
                """,
                "relevant",
                [True],
                id="single-irrelevant-enable-next-line",
            ),
            pytest.param(
                """\
                + # rapids-pre-commit-hooks: enable-next-line rapids-pre-commit-hooks: disable-next-line rapids-pre-commit-hooks: enable
                : ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~0
                :                                                                                        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~1
                + # rapids-pre-commit-hooks: enable rapids-pre-commit-hooks: disable
                : ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~2
                :                                                                   ~3
                + # rapids-pre-commit-hooks: enable
                : ~~3
                :   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~4
                """,  # noqa: E501
                "test",
                [True, True, False, False, True],
                id="complex-next-line",
            ),
            pytest.param(
                """\
                + # prapids-pre-commit-hooks: enable
                : ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~0
                + # rapids-pre-commit-hooks: enabled
                : ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~0
                + # rapids-pre-commit-hooks: enable-next-lines
                : ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~0
                + Hello
                : ~~~~~~0
                """,
                "test",
                [True],
                id="invalid-directives",
            ),
            pytest.param(
                """\
                + # rapids-pre-commit-hooks: disable-next-line
                : ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~0
                + Hello
                : ~~~~~1
                :      ~2
                + # rapids-pre-commit-hooks: disable-next-line
                : ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~2
                + Hello
                : ~~~~~3
                :      ~4
                """,
                "test",
                [True, False, True, False, True],
                id="multiple-next-line",
            ),
        ],
    )
    def test_get_disabled_enabled_boundaries(
        self, content, warning_name, expected_boundaries
    ):
        content, spans = parse_named_spans(content)
        assert Linter.get_disabled_enabled_boundaries(
            Lines(content), warning_name
        ) == list(zip(spans, expected_boundaries, strict=True))

    @pytest.mark.parametrize(
        ["content", "expected_enabled"],
        [
            pytest.param(
                ": ^warning",
                True,
                id="empty-content",
            ),
            pytest.param(
                """\
                + Hello
                : ^warning
                """,
                True,
                id="empty-span-at-start",
            ),
            pytest.param(
                """\
                + Hello
                :    ^warning
                """,
                True,
                id="empty-span-in-middle",
            ),
            pytest.param(
                """\
                + Hello
                :       ^warning
                """,
                True,
                id="empty-span-at-end",
            ),
            pytest.param(
                """\
                + Hello
                : ~~~~~warning
                """,
                True,
                id="no-directives",
            ),
            pytest.param(
                """\
                + # rapids-pre-commit-hooks: enable
                + # rapids-pre-commit-hooks: disable
                :   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~warning
                + # rapids-pre-commit-hooks: enable
                : ~~warning
                """,
                False,
                id="boundary-inside",
            ),
            pytest.param(
                """\
                + # rapids-pre-commit-hooks: enable
                + # rapids-pre-commit-hooks: disable
                :    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~warning
                + # rapids-pre-commit-hooks: enable
                : ~warning
                """,
                False,
                id="boundary-fully-inside",
            ),
            pytest.param(
                """\
                + # rapids-pre-commit-hooks: enable
                + # rapids-pre-commit-hooks: disable
                :  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~warning
                + # rapids-pre-commit-hooks: enable
                : ~~warning
                """,
                True,
                id="boundary-left",
            ),
            pytest.param(
                """\
                + # rapids-pre-commit-hooks: enable
                + # rapids-pre-commit-hooks: disable
                :   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~warning
                + # rapids-pre-commit-hooks: enable
                : ~~~warning
                """,
                True,
                id="boundary-right",
            ),
        ],
    )
    def test_is_warning_span_enabled(self, content, expected_enabled):
        content, spans = parse_named_spans(content)
        boundaries = Linter.get_disabled_enabled_boundaries(
            Lines(content), "relevant"
        )
        assert (
            Linter.is_warning_span_enabled(boundaries, spans["warning"])
            == expected_enabled
        )


class TestLintMain:
    def span_to_line_col(self, content, span):
        lines = Lines(content)
        pos, _ = span
        line = lines.line_for_pos(pos)
        line_span = lines.spans[line]
        return f"{line + 1}:{span[0] - line_span[0] + 1}"

    @pytest.fixture
    def file_with_spans(self, tmp_path):
        with contextlib.ExitStack() as stack:

            def factory(filename, contents):
                contents, spans = parse_named_spans(contents)
                f = stack.enter_context(
                    open(os.path.join(tmp_path, filename), "w+")
                )
                f.write(contents)
                f.flush()
                f.seek(0)
                return f, contents, spans

            yield factory

    @pytest.fixture
    def hello_world_file(self, file_with_spans):
        return file_with_spans(
            "hello_world.txt",
            """\
            > Hello world!
            : ~~~~~greeting
            :      ^punctuation
            :       ~~~~~world
            """,
        )

    @pytest.fixture
    def hello_file(self, file_with_spans):
        return file_with_spans(
            "hello.txt",
            """\
            > Hello!
            : ~~~~~greeting
            :      ^punctuation
            """,
        )

    @pytest.fixture
    def binary_file(self, tmp_path):
        with open(os.path.join(tmp_path, "binary.bin"), "wb+") as f:
            f.write(b"\xde\xad\xbe\xef")
            f.flush()
            f.seek(0)
            yield f

    @pytest.fixture
    def long_file(self, file_with_spans):
        return file_with_spans(
            "long.txt",
            """\
            + This is a long file
            : ~~~~~~~~~~~~~~~~~~~first_line
            : ~~~~~~~~~~~~~~~~~~~~whole_file
            + It has multiple lines
            : ~~~~~~~~~~~~~~~~~~~~~~whole_file
            """,
        )

    @pytest.fixture
    def bracket_file(self, file_with_spans):
        return file_with_spans(
            "file[with]brackets.txt",
            """\
            + This [file] [has] [brackets]
            : ~~~~~~~~~~~~~~~~~~~~~~~~~~~~warning
            :             ~~~~~replacement
            """,
        )

    @pytest.fixture
    def disabled_file(self, file_with_spans):
        return file_with_spans(
            "disabled.txt",
            """\
            + # rapids-pre-commit-hooks: disable
            + Hello!
            :      ~shout
            """,
        )

    @contextlib.contextmanager
    def mock_console(self):
        m = Mock()
        with (
            patch("rich.console.Console", m),
            patch("rapids_pre_commit_hooks.lint.Console", m),
        ):
            yield m

    @pytest.fixture
    def hello_check(self, hello_file, hello_world_file):
        _, hello_file_content, hello_file_spans = hello_file
        _, hello_world_file_content, hello_world_file_spans = hello_world_file

        def check(linter, args):
            assert args.check_test
            if linter.content == hello_file_content:
                w = linter.add_warning(
                    hello_file_spans["greeting"], "say good bye instead"
                )
                w.add_replacement(hello_file_spans["greeting"], "Good bye")
            elif linter.content == hello_world_file_content:
                w = linter.add_warning(
                    hello_world_file_spans["greeting"], "say good bye instead"
                )
                w.add_replacement(
                    hello_world_file_spans["greeting"], "Good bye"
                )
                if args.check_test_note:
                    w.add_note(
                        hello_world_file_spans["world"],
                        "it's a small world after all",
                    )
                linter.add_warning(
                    hello_world_file_spans["punctuation"], "use punctuation"
                ).add_replacement(hello_world_file_spans["punctuation"], ",")

        return check

    @pytest.fixture
    def long_file_check(self, long_file):
        _, _, spans = long_file

        def check(linter, _args):
            linter.add_warning(spans["whole_file"], "this is a long file")

        return check

    @pytest.fixture
    def long_fix_check(self, long_file):
        _, _, spans = long_file

        def check(linter, _args):
            linter.add_warning(
                spans["first_line"], "this is a long line"
            ).add_replacement(
                spans["first_line"],
                "This is a long file\nIt's even longer now",
            )

        return check

    @pytest.fixture
    def long_delete_fix_check(self, long_file):
        _, _, spans = long_file

        def check(linter, _args):
            linter.add_warning(
                spans["whole_file"], "this is a long file"
            ).add_replacement(spans["whole_file"], "This is a short file now")

        return check

    @pytest.fixture
    def bracket_check(self, bracket_file):
        _, _, spans = bracket_file

        def check(linter, _args):
            linter.add_warning(
                spans["warning"], "this [file] has brackets"
            ).add_replacement(spans["replacement"], "[has more]")

        return check

    @pytest.fixture
    def disabled_check(self, disabled_file):
        _, _, spans = disabled_file

        def check(linter, _args):
            linter.add_warning(spans["shout"], "don't shout")

        return check

    def test_no_warnings_no_fix(self, hello_world_file):
        file, _, _ = hello_world_file
        with (
            patch(
                "sys.argv",
                ["check-test", "--check-test", file.name],
            ),
            self.mock_console() as console,
        ):
            m = LintMain("test")
            m.argparser.add_argument("--check-test", action="store_true")
            m.argparser.add_argument("--check-test-note", action="store_true")
            with m.execute():
                pass
        assert file.read() == "Hello world!"
        assert console.mock_calls == [
            call(highlight=False),
        ]

    def test_no_warnings_fix(self, hello_world_file):
        file, _, _ = hello_world_file
        with (
            patch(
                "sys.argv",
                ["check-test", "--check-test", "--fix", file.name],
            ),
            self.mock_console() as console,
        ):
            m = LintMain("test")
            m.argparser.add_argument("--check-test", action="store_true")
            m.argparser.add_argument("--check-test-note", action="store_true")
            with m.execute():
                pass
        assert file.read() == "Hello world!"
        assert console.mock_calls == [
            call(highlight=False),
        ]

    def test_warnings_no_fix(self, hello_world_file, hello_check):
        file, content, spans = hello_world_file
        with (
            patch(
                "sys.argv",
                ["check-test", "--check-test", file.name],
            ),
            self.mock_console() as console,
            pytest.raises(SystemExit, match=r"^1$"),
        ):
            m = LintMain("test")
            m.argparser.add_argument("--check-test", action="store_true")
            m.argparser.add_argument("--check-test-note", action="store_true")
            with m.execute() as ctx:
                ctx.add_check(hello_check)
        assert file.read() == "Hello world!"
        assert console.mock_calls == [
            call(highlight=False),
            call().print(
                f"In file [bold]{file.name}:"
                f"{self.span_to_line_col(content, spans['greeting'])}"
                "[/bold]:"
            ),
            call().print(" [bold]Hello[/bold] world!"),
            call().print("[bold]warning:[/bold] say good bye instead"),
            call().print(),
            call().print(
                f"In file [bold]{file.name}:"
                f"{self.span_to_line_col(content, spans['greeting'])}"
                "[/bold]:"
            ),
            call().print("[red]-[bold]Hello[/bold] world![/red]"),
            call().print("[green]+[bold]Good bye[/bold] world![/green]"),
            call().print("[bold]note:[/bold] suggested fix"),
            call().print(),
            call().print(
                f"In file [bold]{file.name}:"
                f"{self.span_to_line_col(content, spans['punctuation'])}"
                "[/bold]:"
            ),
            call().print(" Hello[bold][/bold] world!"),
            call().print("[bold]warning:[/bold] use punctuation"),
            call().print(),
            call().print(
                f"In file [bold]{file.name}:"
                f"{self.span_to_line_col(content, spans['punctuation'])}"
                "[/bold]:"
            ),
            call().print("[red]-Hello[bold][/bold] world![/red]"),
            call().print("[green]+Hello[bold],[/bold] world![/green]"),
            call().print("[bold]note:[/bold] suggested fix"),
            call().print(),
        ]

    def test_warnings_fix(self, hello_world_file, hello_check):
        file, content, spans = hello_world_file
        with (
            patch(
                "sys.argv",
                ["check-test", "--check-test", "--fix", file.name],
            ),
            self.mock_console() as console,
            pytest.raises(SystemExit, match=r"^1$"),
        ):
            m = LintMain("test")
            m.argparser.add_argument("--check-test", action="store_true")
            m.argparser.add_argument("--check-test-note", action="store_true")
            with m.execute() as ctx:
                ctx.add_check(hello_check)
        assert file.read() == "Good bye, world!"
        assert console.mock_calls == [
            call(highlight=False),
            call().print(
                f"In file [bold]{file.name}:"
                f"{self.span_to_line_col(content, spans['greeting'])}"
                "[/bold]:"
            ),
            call().print(" [bold]Hello[/bold] world!"),
            call().print("[bold]warning:[/bold] say good bye instead"),
            call().print(),
            call().print(
                f"In file [bold]{file.name}:"
                f"{self.span_to_line_col(content, spans['greeting'])}"
                "[/bold]:"
            ),
            call().print("[red]-[bold]Hello[/bold] world![/red]"),
            call().print("[green]+[bold]Good bye[/bold] world![/green]"),
            call().print("[bold]note:[/bold] suggested fix applied"),
            call().print(),
            call().print(
                f"In file [bold]{file.name}:"
                f"{self.span_to_line_col(content, spans['punctuation'])}"
                "[/bold]:"
            ),
            call().print(" Hello[bold][/bold] world!"),
            call().print("[bold]warning:[/bold] use punctuation"),
            call().print(),
            call().print(
                f"In file [bold]{file.name}:"
                f"{self.span_to_line_col(content, spans['punctuation'])}"
                "[/bold]:"
            ),
            call().print("[red]-Hello[bold][/bold] world![/red]"),
            call().print("[green]+Hello[bold],[/bold] world![/green]"),
            call().print("[bold]note:[/bold] suggested fix applied"),
            call().print(),
        ]

    def test_warnings_note(self, hello_world_file, hello_check):
        file, content, spans = hello_world_file
        with (
            patch(
                "sys.argv",
                [
                    "check-test",
                    "--check-test",
                    "--check-test-note",
                    file.name,
                ],
            ),
            self.mock_console() as console,
            pytest.raises(SystemExit, match=r"^1$"),
        ):
            m = LintMain("test")
            m.argparser.add_argument("--check-test", action="store_true")
            m.argparser.add_argument("--check-test-note", action="store_true")
            with m.execute() as ctx:
                ctx.add_check(hello_check)
        assert file.read() == "Hello world!"
        assert console.mock_calls == [
            call(highlight=False),
            call().print(
                f"In file [bold]{file.name}:"
                f"{self.span_to_line_col(content, spans['greeting'])}"
                "[/bold]:"
            ),
            call().print(" [bold]Hello[/bold] world!"),
            call().print("[bold]warning:[/bold] say good bye instead"),
            call().print(),
            call().print(
                f"In file [bold]{file.name}:"
                f"{self.span_to_line_col(content, spans['world'])}"
                "[/bold]:"
            ),
            call().print(" Hello [bold]world[/bold]!"),
            call().print("[bold]note:[/bold] it's a small world after all"),
            call().print(),
            call().print(
                f"In file [bold]{file.name}:"
                f"{self.span_to_line_col(content, spans['greeting'])}"
                "[/bold]:"
            ),
            call().print("[red]-[bold]Hello[/bold] world![/red]"),
            call().print("[green]+[bold]Good bye[/bold] world![/green]"),
            call().print("[bold]note:[/bold] suggested fix"),
            call().print(),
            call().print(
                f"In file [bold]{file.name}:"
                f"{self.span_to_line_col(content, spans['punctuation'])}"
                "[/bold]:"
            ),
            call().print(" Hello[bold][/bold] world!"),
            call().print("[bold]warning:[/bold] use punctuation"),
            call().print(),
            call().print(
                f"In file [bold]{file.name}:"
                f"{self.span_to_line_col(content, spans['punctuation'])}"
                "[/bold]:"
            ),
            call().print("[red]-Hello[bold][/bold] world![/red]"),
            call().print("[green]+Hello[bold],[/bold] world![/green]"),
            call().print("[bold]note:[/bold] suggested fix"),
            call().print(),
        ]

    def test_multiple_files(self, hello_world_file, hello_file, hello_check):
        hello_file, hello_content, hello_spans = hello_file
        hello_world_file, world_content, world_spans = hello_world_file
        with (
            patch(
                "sys.argv",
                [
                    "check-test",
                    "--check-test",
                    "--fix",
                    hello_world_file.name,
                    hello_file.name,
                ],
            ),
            self.mock_console() as console,
            pytest.raises(SystemExit, match=r"^1$"),
        ):
            m = LintMain("test")
            m.argparser.add_argument("--check-test", action="store_true")
            m.argparser.add_argument("--check-test-note", action="store_true")
            with m.execute() as ctx:
                ctx.add_check(hello_check)
        assert hello_world_file.read() == "Good bye, world!"
        assert hello_file.read() == "Good bye!"
        assert console.mock_calls == [
            call(highlight=False),
            call().print(
                f"In file [bold]{hello_world_file.name}:"
                f"{self.span_to_line_col(world_content, world_spans['greeting'])}"  # noqa: E501
                "[/bold]:"
            ),
            call().print(" [bold]Hello[/bold] world!"),
            call().print("[bold]warning:[/bold] say good bye instead"),
            call().print(),
            call().print(
                f"In file [bold]{hello_world_file.name}:"
                f"{self.span_to_line_col(world_content, world_spans['greeting'])}"  # noqa: E501
                "[/bold]:"
            ),
            call().print("[red]-[bold]Hello[/bold] world![/red]"),
            call().print("[green]+[bold]Good bye[/bold] world![/green]"),
            call().print("[bold]note:[/bold] suggested fix applied"),
            call().print(),
            call().print(
                f"In file [bold]{hello_world_file.name}:"
                f"{self.span_to_line_col(world_content, world_spans['punctuation'])}"  # noqa: E501
                "[/bold]:"
            ),
            call().print(" Hello[bold][/bold] world!"),
            call().print("[bold]warning:[/bold] use punctuation"),
            call().print(),
            call().print(
                f"In file [bold]{hello_world_file.name}:"
                f"{self.span_to_line_col(world_content, world_spans['punctuation'])}"  # noqa: E501
                "[/bold]:"
            ),
            call().print("[red]-Hello[bold][/bold] world![/red]"),
            call().print("[green]+Hello[bold],[/bold] world![/green]"),
            call().print("[bold]note:[/bold] suggested fix applied"),
            call().print(),
            call(highlight=False),
            call().print(
                f"In file [bold]{hello_file.name}:"
                f"{self.span_to_line_col(hello_content, hello_spans['greeting'])}"  # noqa: E501
                "[/bold]:"
            ),
            call().print(" [bold]Hello[/bold]!"),
            call().print("[bold]warning:[/bold] say good bye instead"),
            call().print(),
            call().print(
                f"In file [bold]{hello_file.name}:"
                f"{self.span_to_line_col(hello_content, hello_spans['greeting'])}"  # noqa: E501
                "[/bold]:"
            ),
            call().print("[red]-[bold]Hello[/bold]![/red]"),
            call().print("[green]+[bold]Good bye[/bold]![/green]"),
            call().print("[bold]note:[/bold] suggested fix applied"),
            call().print(),
        ]

    @pytest.mark.parametrize(
        "content",
        [
            "\n",
            "\r\n",
            "\r",
        ],
    )
    def test_newline_type(self, tmp_path, content):
        with open(os.path.join(tmp_path, "file.txt"), "w") as f:
            f.write(content)

        the_check = Mock()

        with (
            patch(
                "sys.argv",
                [
                    "check-test",
                    os.path.join(tmp_path, "file.txt"),
                ],
            ),
        ):
            m = LintMain("test")
            with m.execute() as ctx:
                ctx.add_check(the_check)
        the_check.assert_called_once()
        assert the_check.call_args[0][0].content == content
        assert the_check.call_args[0][0].lines.newline_style == content

    def test_binary_file(self, binary_file, hello_check):
        file = binary_file
        mock_linter = Mock(wraps=Linter)
        with (
            patch(
                "sys.argv",
                [
                    "check-test",
                    "--check-test",
                    "--fix",
                    file.name,
                ],
            ),
            patch("rapids_pre_commit_hooks.lint.Linter", mock_linter),
            pytest.warns(
                BinaryFileWarning,
                match=r"^Refusing to run text linter on binary file .*\.$",
            ),
        ):
            m = LintMain("test")
            m.argparser.add_argument("--check-test", action="store_true")
            m.argparser.add_argument("--check-test-note", action="store_true")
            with m.execute() as ctx:
                ctx.add_check(hello_check)
        mock_linter.assert_not_called()

    def test_long_file(self, long_file, long_file_check, long_fix_check):
        file, content, spans = long_file
        with (
            patch(
                "sys.argv",
                [
                    "check-test",
                    file.name,
                ],
            ),
            self.mock_console() as console,
            pytest.raises(SystemExit, match=r"^1$"),
        ):
            m = LintMain("test")
            with m.execute() as ctx:
                ctx.add_check(long_file_check)
                ctx.add_check(long_fix_check)
        assert file.read() == dedent(
            """\
            This is a long file
            It has multiple lines
            """
        )
        assert console.mock_calls == [
            call(highlight=False),
            call().print(
                f"In file [bold]{file.name}:"
                f"{self.span_to_line_col(content, spans['first_line'])}"
                "[/bold]:"
            ),
            call().print(" [bold]This is a long file[/bold]"),
            call().print("[bold]warning:[/bold] this is a long line"),
            call().print(),
            call().print(
                f"In file [bold]{file.name}:"
                f"{self.span_to_line_col(content, spans['first_line'])}"
                "[/bold]:"
            ),
            call().print("[red]-[bold]This is a long file[/bold][/red]"),
            call().print("[green]+[bold]This is a long file[/bold][/green]"),
            call().print(
                "[bold]note:[/bold] suggested fix is too long to display, use "
                "--fix to apply it"
            ),
            call().print(),
            call().print(
                f"In file [bold]{file.name}:"
                f"{self.span_to_line_col(content, spans['whole_file'])}"
                "[/bold]:"
            ),
            call().print(" [bold]This is a long file[/bold]"),
            call().print("[bold]warning:[/bold] this is a long file"),
            call().print(),
        ]

    def test_long_file_delete(self, long_file, long_delete_fix_check):
        file, content, spans = long_file
        with (
            patch(
                "sys.argv",
                [
                    "check-test",
                    file.name,
                ],
            ),
            self.mock_console() as console,
            pytest.raises(SystemExit, match=r"^1$"),
        ):
            m = LintMain("test")
            with m.execute() as ctx:
                ctx.add_check(long_delete_fix_check)
        assert file.read() == dedent(
            """\
            This is a long file
            It has multiple lines
            """
        )
        assert console.mock_calls == [
            call(highlight=False),
            call().print(
                f"In file [bold]{file.name}:"
                f"{self.span_to_line_col(content, spans['whole_file'])}"
                "[/bold]:"
            ),
            call().print(" [bold]This is a long file[/bold]"),
            call().print("[bold]warning:[/bold] this is a long file"),
            call().print(),
            call().print(
                f"In file [bold]{file.name}:"
                f"{self.span_to_line_col(content, spans['whole_file'])}"
                "[/bold]:"
            ),
            call().print("[red]-[bold]This is a long file[/bold][/red]"),
            call().print(
                "[green]+[bold]This is a short file now[/bold][/green]"
            ),
            call().print(
                "[bold]note:[/bold] suggested fix is too long to display, use "
                "--fix to apply it"
            ),
            call().print(),
        ]

    def test_long_file_fix(self, long_file, long_file_check, long_fix_check):
        file, content, spans = long_file
        with (
            patch(
                "sys.argv",
                [
                    "check-test",
                    "--fix",
                    file.name,
                ],
            ),
            self.mock_console() as console,
            pytest.raises(SystemExit, match=r"^1$"),
        ):
            m = LintMain("test")
            with m.execute() as ctx:
                ctx.add_check(long_file_check)
                ctx.add_check(long_fix_check)
        assert file.read() == dedent(
            """\
            This is a long file
            It's even longer now
            It has multiple lines
            """
        )
        assert console.mock_calls == [
            call(highlight=False),
            call().print(
                f"In file [bold]{file.name}:"
                f"{self.span_to_line_col(content, spans['first_line'])}"
                "[/bold]:"
            ),
            call().print(" [bold]This is a long file[/bold]"),
            call().print("[bold]warning:[/bold] this is a long line"),
            call().print(),
            call().print(
                f"In file [bold]{file.name}:"
                f"{self.span_to_line_col(content, spans['first_line'])}"
                "[/bold]:"
            ),
            call().print("[red]-[bold]This is a long file[/bold][/red]"),
            call().print("[green]+[bold]This is a long file[/bold][/green]"),
            call().print(
                "[bold]note:[/bold] suggested fix applied but is too long to "
                "display"
            ),
            call().print(),
            call().print(
                f"In file [bold]{file.name}:"
                f"{self.span_to_line_col(content, spans['whole_file'])}"
                "[/bold]:"
            ),
            call().print(" [bold]This is a long file[/bold]"),
            call().print("[bold]warning:[/bold] this is a long file"),
            call().print(),
        ]

    def test_long_file_delete_fix(self, long_file, long_delete_fix_check):
        file, content, spans = long_file
        with (
            patch(
                "sys.argv",
                [
                    "check-test",
                    "--fix",
                    file.name,
                ],
            ),
            self.mock_console() as console,
            pytest.raises(SystemExit, match=r"^1$"),
        ):
            m = LintMain("test")
            with m.execute() as ctx:
                ctx.add_check(long_delete_fix_check)
        assert file.read() == "This is a short file now"
        assert console.mock_calls == [
            call(highlight=False),
            call().print(
                f"In file [bold]{file.name}:"
                f"{self.span_to_line_col(content, spans['whole_file'])}"
                "[/bold]:"
            ),
            call().print(" [bold]This is a long file[/bold]"),
            call().print("[bold]warning:[/bold] this is a long file"),
            call().print(),
            call().print(
                f"In file [bold]{file.name}:"
                f"{self.span_to_line_col(content, spans['whole_file'])}"
                "[/bold]:"
            ),
            call().print("[red]-[bold]This is a long file[/bold][/red]"),
            call().print(
                "[green]+[bold]This is a short file now[/bold][/green]"
            ),
            call().print(
                "[bold]note:[/bold] suggested fix applied but is too long to "
                "display"
            ),
            call().print(),
        ]

    def test_bracket_file(self, bracket_file, bracket_check):
        file, content, spans = bracket_file
        with (
            patch(
                "sys.argv",
                [
                    "check-test",
                    "--fix",
                    file.name,
                ],
            ),
            self.mock_console() as console,
            pytest.raises(SystemExit, match=r"^1$"),
        ):
            m = LintMain("test")
            with m.execute() as ctx:
                ctx.add_check(bracket_check)
        assert file.read() == "This [file] [has more] [brackets]\n"
        assert console.mock_calls == [
            call(highlight=False),
            call().print(
                rf"In file [bold]{os.path.dirname(file.name)}"
                r"/file\[with]brackets.txt:"
                f"{self.span_to_line_col(content, spans['warning'])}"
                "[/bold]:"
            ),
            call().print(r" [bold]This \[file] \[has] \[brackets][/bold]"),
            call().print(r"[bold]warning:[/bold] this \[file] has brackets"),
            call().print(),
            call().print(
                rf"In file [bold]{os.path.dirname(file.name)}"
                r"/file\[with]brackets.txt:"
                f"{self.span_to_line_col(content, spans['replacement'])}"
                "[/bold]:"
            ),
            call().print(
                r"[red]-This \[file] [bold]\[has][/bold] \[brackets][/red]"
            ),
            call().print(
                r"[green]+This \[file] [bold]\[has more][/bold] \[brackets]"
                r"[/green]"
            ),
            call().print("[bold]note:[/bold] suggested fix applied"),
            call().print(),
        ]

    def test_disabled_file(self, disabled_file, disabled_check):
        file, content, _ = disabled_file

        with (
            patch(
                "sys.argv",
                [
                    "check-test",
                    "--fix",
                    file.name,
                ],
            ),
            self.mock_console() as console,
        ):
            m = LintMain("test")
            with m.execute() as ctx:
                ctx.add_check(disabled_check)
        assert file.read() == content
        assert console.mock_calls == [
            call(highlight=False),
        ]
