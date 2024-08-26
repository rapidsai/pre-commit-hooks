# Copyright (c) 2024, NVIDIA CORPORATION.
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

import contextlib
import os.path
from textwrap import dedent
from unittest.mock import Mock, call, patch

import pytest

from rapids_pre_commit_hooks.lint import (
    BinaryFileWarning,
    Linter,
    LintMain,
    OverlappingReplacementsError,
)


class TestLinter:
    LONG_CONTENTS = (
        "line 1\nline 2\rline 3\r\nline 4\r\n\nline 6\r\n\r\nline 8\n\r\n"
        "line 10\r\r\nline 12\r\n\rline 14\n\nline 16\r\rline 18\n\rline 20"
    )

    def test_lines(self):
        linter = Linter("test.txt", self.LONG_CONTENTS)
        assert linter.lines == [
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
        ]

        linter = Linter("test.txt", "line 1\n")
        assert linter.lines == [
            (0, 6),
            (7, 7),
        ]

        linter = Linter("test.txt", "line 1\r\n")
        assert linter.lines == [
            (0, 6),
            (8, 8),
        ]

        linter = Linter("test.txt", "")
        assert linter.lines == [
            (0, 0),
        ]

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
                    IndexError, match="^Position 21 is inside a line separator$"
                ),
            ),
            (LONG_CONTENTS, 34, 5, contextlib.nullcontext()),
            (LONG_CONTENTS, 97, 19, contextlib.nullcontext()),
            (LONG_CONTENTS, 104, 19, contextlib.nullcontext()),
            (
                LONG_CONTENTS,
                200,
                None,
                pytest.raises(IndexError, match="^Position 200 is not in the string$"),
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
        linter = Linter("test.txt", contents)
        with raises:
            assert linter.line_for_pos(pos) == line

    def test_fix(self):
        linter = Linter("test.txt", "Hello world!")
        assert linter.fix() == "Hello world!"

        linter.add_warning((0, 0), "no fix")
        assert linter.fix() == "Hello world!"

        linter.add_warning((5, 5), "use punctuation").add_replacement((5, 5), ",")
        linter.add_warning((0, 5), "say good bye instead").add_replacement(
            (0, 5), "Good bye"
        )
        linter.add_warning((11, 12), "don't shout").add_replacement((11, 12), "")
        linter.add_warning((6, 11), "no-op replacement").add_replacement((11, 11), "")
        assert linter.fix() == "Good bye, world"

        linter.add_warning((11, 12), "don't shout").add_replacement((11, 12), ".")
        with pytest.raises(
            OverlappingReplacementsError,
            match=r"^Replacement\(pos=\(11, 12\), newtext=''\) overlaps with "
            + r"Replacement\(pos=\(11, 12\), newtext='\.'\)$",
        ):
            linter.fix()


class TestLintMain:
    @pytest.fixture
    def hello_world_file(self, tmp_path):
        with open(os.path.join(tmp_path, "hello_world.txt"), "w+") as f:
            f.write("Hello world!")
            f.flush()
            f.seek(0)
            yield f

    @pytest.fixture
    def hello_file(self, tmp_path):
        with open(os.path.join(tmp_path, "hello.txt"), "w+") as f:
            f.write("Hello!")
            f.flush()
            f.seek(0)
            yield f

    @pytest.fixture
    def binary_file(self, tmp_path):
        with open(os.path.join(tmp_path, "binary.bin"), "wb+") as f:
            f.write(b"\xDE\xAD\xBE\xEF")
            f.flush()
            f.seek(0)
            yield f

    @pytest.fixture
    def long_file(self, tmp_path):
        with open(os.path.join(tmp_path, "long.txt"), "w+") as f:
            f.write("This is a long file\nIt has multiple lines\n")
            f.flush()
            f.seek(0)
            yield f

    @pytest.fixture
    def bracket_file(self, tmp_path):
        with open(os.path.join(tmp_path, "file[with]brackets.txt"), "w+") as f:
            f.write("This [file] [has] [brackets]\n")
            f.flush()
            f.seek(0)
            yield f

    @contextlib.contextmanager
    def mock_console(self):
        m = Mock()
        with patch("rich.console.Console", m), patch(
            "rapids_pre_commit_hooks.lint.Console", m
        ):
            yield m

    def the_check(self, linter, args):
        assert args.check_test
        linter.add_warning((0, 5), "say good bye instead").add_replacement(
            (0, 5), "Good bye"
        )
        if linter.content[5] != "!":
            linter.add_warning((5, 5), "use punctuation").add_replacement((5, 5), ",")

    def long_file_check(self, linter, args):
        linter.add_warning((0, len(linter.content)), "this is a long file")

    def long_fix_check(self, linter, args):
        linter.add_warning((0, 19), "this is a long line").add_replacement(
            (0, 19), "This is a long file\nIt's even longer now"
        )

    def long_delete_fix_check(self, linter, args):
        linter.add_warning(
            (0, len(linter.content)), "this is a long file"
        ).add_replacement((0, len(linter.content)), "This is a short file now")

    def bracket_check(self, linter, args):
        linter.add_warning((0, 28), "this [file] has brackets").add_replacement(
            (12, 17), "[has more]"
        )

    def test_no_warnings_no_fix(self, hello_world_file):
        with patch(
            "sys.argv", ["check-test", "--check-test", hello_world_file.name]
        ), self.mock_console() as console:
            m = LintMain()
            m.argparser.add_argument("--check-test", action="store_true")
            with m.execute():
                pass
        assert hello_world_file.read() == "Hello world!"
        assert console.mock_calls == [
            call(highlight=False),
        ]

    def test_no_warnings_fix(self, hello_world_file):
        with patch(
            "sys.argv", ["check-test", "--check-test", "--fix", hello_world_file.name]
        ), self.mock_console() as console:
            m = LintMain()
            m.argparser.add_argument("--check-test", action="store_true")
            with m.execute():
                pass
        assert hello_world_file.read() == "Hello world!"
        assert console.mock_calls == [
            call(highlight=False),
        ]

    def test_warnings_no_fix(self, hello_world_file):
        with patch(
            "sys.argv", ["check-test", "--check-test", hello_world_file.name]
        ), self.mock_console() as console, pytest.raises(SystemExit, match=r"^1$"):
            m = LintMain()
            m.argparser.add_argument("--check-test", action="store_true")
            with m.execute() as ctx:
                ctx.add_check(self.the_check)
        assert hello_world_file.read() == "Hello world!"
        assert console.mock_calls == [
            call(highlight=False),
            call().print(f"In file [bold]{hello_world_file.name}:1:1[/bold]:"),
            call().print(" [bold]Hello[/bold] world!"),
            call().print("[bold]warning:[/bold] say good bye instead"),
            call().print(),
            call().print(f"In file [bold]{hello_world_file.name}:1:1[/bold]:"),
            call().print("[red]-[bold]Hello[/bold] world![/red]"),
            call().print("[green]+[bold]Good bye[/bold] world![/green]"),
            call().print("[bold]note:[/bold] suggested fix"),
            call().print(),
            call().print(f"In file [bold]{hello_world_file.name}:1:6[/bold]:"),
            call().print(" Hello[bold][/bold] world!"),
            call().print("[bold]warning:[/bold] use punctuation"),
            call().print(),
            call().print(f"In file [bold]{hello_world_file.name}:1:6[/bold]:"),
            call().print("[red]-Hello[bold][/bold] world![/red]"),
            call().print("[green]+Hello[bold],[/bold] world![/green]"),
            call().print("[bold]note:[/bold] suggested fix"),
            call().print(),
        ]

    def test_warnings_fix(self, hello_world_file):
        with patch(
            "sys.argv", ["check-test", "--check-test", "--fix", hello_world_file.name]
        ), self.mock_console() as console, pytest.raises(SystemExit, match=r"^1$"):
            m = LintMain()
            m.argparser.add_argument("--check-test", action="store_true")
            with m.execute() as ctx:
                ctx.add_check(self.the_check)
        assert hello_world_file.read() == "Good bye, world!"
        assert console.mock_calls == [
            call(highlight=False),
            call().print(f"In file [bold]{hello_world_file.name}:1:1[/bold]:"),
            call().print(" [bold]Hello[/bold] world!"),
            call().print("[bold]warning:[/bold] say good bye instead"),
            call().print(),
            call().print(f"In file [bold]{hello_world_file.name}:1:1[/bold]:"),
            call().print("[red]-[bold]Hello[/bold] world![/red]"),
            call().print("[green]+[bold]Good bye[/bold] world![/green]"),
            call().print("[bold]note:[/bold] suggested fix applied"),
            call().print(),
            call().print(f"In file [bold]{hello_world_file.name}:1:6[/bold]:"),
            call().print(" Hello[bold][/bold] world!"),
            call().print("[bold]warning:[/bold] use punctuation"),
            call().print(),
            call().print(f"In file [bold]{hello_world_file.name}:1:6[/bold]:"),
            call().print("[red]-Hello[bold][/bold] world![/red]"),
            call().print("[green]+Hello[bold],[/bold] world![/green]"),
            call().print("[bold]note:[/bold] suggested fix applied"),
            call().print(),
        ]

    def test_multiple_files(self, hello_world_file, hello_file):
        with patch(
            "sys.argv",
            [
                "check-test",
                "--check-test",
                "--fix",
                hello_world_file.name,
                hello_file.name,
            ],
        ), self.mock_console() as console, pytest.raises(SystemExit, match=r"^1$"):
            m = LintMain()
            m.argparser.add_argument("--check-test", action="store_true")
            with m.execute() as ctx:
                ctx.add_check(self.the_check)
        assert hello_world_file.read() == "Good bye, world!"
        assert hello_file.read() == "Good bye!"
        assert console.mock_calls == [
            call(highlight=False),
            call().print(f"In file [bold]{hello_world_file.name}:1:1[/bold]:"),
            call().print(" [bold]Hello[/bold] world!"),
            call().print("[bold]warning:[/bold] say good bye instead"),
            call().print(),
            call().print(f"In file [bold]{hello_world_file.name}:1:1[/bold]:"),
            call().print("[red]-[bold]Hello[/bold] world![/red]"),
            call().print("[green]+[bold]Good bye[/bold] world![/green]"),
            call().print("[bold]note:[/bold] suggested fix applied"),
            call().print(),
            call().print(f"In file [bold]{hello_world_file.name}:1:6[/bold]:"),
            call().print(" Hello[bold][/bold] world!"),
            call().print("[bold]warning:[/bold] use punctuation"),
            call().print(),
            call().print(f"In file [bold]{hello_world_file.name}:1:6[/bold]:"),
            call().print("[red]-Hello[bold][/bold] world![/red]"),
            call().print("[green]+Hello[bold],[/bold] world![/green]"),
            call().print("[bold]note:[/bold] suggested fix applied"),
            call().print(),
            call(highlight=False),
            call().print(f"In file [bold]{hello_file.name}:1:1[/bold]:"),
            call().print(" [bold]Hello[/bold]!"),
            call().print("[bold]warning:[/bold] say good bye instead"),
            call().print(),
            call().print(f"In file [bold]{hello_file.name}:1:1[/bold]:"),
            call().print("[red]-[bold]Hello[/bold]![/red]"),
            call().print("[green]+[bold]Good bye[/bold]![/green]"),
            call().print("[bold]note:[/bold] suggested fix applied"),
            call().print(),
        ]

    def test_binary_file(self, binary_file):
        mock_linter = Mock(wraps=Linter)
        with patch(
            "sys.argv",
            [
                "check-test",
                "--check-test",
                "--fix",
                binary_file.name,
            ],
        ), patch("rapids_pre_commit_hooks.lint.Linter", mock_linter), pytest.warns(
            BinaryFileWarning,
            match=r"^Refusing to run text linter on binary file .*\.$",
        ):
            m = LintMain()
            m.argparser.add_argument("--check-test", action="store_true")
            with m.execute() as ctx:
                ctx.add_check(self.the_check)
        mock_linter.assert_not_called()

    def test_long_file(self, long_file):
        with patch(
            "sys.argv",
            [
                "check-test",
                long_file.name,
            ],
        ), self.mock_console() as console, pytest.raises(SystemExit, match=r"^1$"):
            m = LintMain()
            with m.execute() as ctx:
                ctx.add_check(self.long_file_check)
                ctx.add_check(self.long_fix_check)
        assert long_file.read() == dedent(
            """\
            This is a long file
            It has multiple lines
            """
        )
        assert console.mock_calls == [
            call(highlight=False),
            call().print(f"In file [bold]{long_file.name}:1:1[/bold]:"),
            call().print(" [bold]This is a long file[/bold]"),
            call().print("[bold]warning:[/bold] this is a long line"),
            call().print(),
            call().print(f"In file [bold]{long_file.name}:1:1[/bold]:"),
            call().print("[red]-[bold]This is a long file[/bold][/red]"),
            call().print("[green]+[bold]This is a long file[/bold][/green]"),
            call().print(
                "[bold]note:[/bold] suggested fix is too long to display, use --fix to "
                "apply it"
            ),
            call().print(),
            call().print(f"In file [bold]{long_file.name}:1:1[/bold]:"),
            call().print(" [bold]This is a long file[/bold]"),
            call().print("[bold]warning:[/bold] this is a long file"),
            call().print(),
        ]

    def test_long_file_delete(self, long_file):
        with patch(
            "sys.argv",
            [
                "check-test",
                long_file.name,
            ],
        ), self.mock_console() as console, pytest.raises(SystemExit, match=r"^1$"):
            m = LintMain()
            with m.execute() as ctx:
                ctx.add_check(self.long_delete_fix_check)
        assert long_file.read() == dedent(
            """\
            This is a long file
            It has multiple lines
            """
        )
        assert console.mock_calls == [
            call(highlight=False),
            call().print(f"In file [bold]{long_file.name}:1:1[/bold]:"),
            call().print(" [bold]This is a long file[/bold]"),
            call().print("[bold]warning:[/bold] this is a long file"),
            call().print(),
            call().print(f"In file [bold]{long_file.name}:1:1[/bold]:"),
            call().print("[red]-[bold]This is a long file[/bold][/red]"),
            call().print("[green]+[bold]This is a short file now[/bold][/green]"),
            call().print(
                "[bold]note:[/bold] suggested fix is too long to display, use --fix to "
                "apply it"
            ),
            call().print(),
        ]

    def test_long_file_fix(self, long_file):
        with patch(
            "sys.argv",
            [
                "check-test",
                "--fix",
                long_file.name,
            ],
        ), self.mock_console() as console, pytest.raises(SystemExit, match=r"^1$"):
            m = LintMain()
            with m.execute() as ctx:
                ctx.add_check(self.long_file_check)
                ctx.add_check(self.long_fix_check)
        assert long_file.read() == dedent(
            """\
            This is a long file
            It's even longer now
            It has multiple lines
            """
        )
        assert console.mock_calls == [
            call(highlight=False),
            call().print(f"In file [bold]{long_file.name}:1:1[/bold]:"),
            call().print(" [bold]This is a long file[/bold]"),
            call().print("[bold]warning:[/bold] this is a long line"),
            call().print(),
            call().print(f"In file [bold]{long_file.name}:1:1[/bold]:"),
            call().print("[red]-[bold]This is a long file[/bold][/red]"),
            call().print("[green]+[bold]This is a long file[/bold][/green]"),
            call().print(
                "[bold]note:[/bold] suggested fix applied but is too long to display"
            ),
            call().print(),
            call().print(f"In file [bold]{long_file.name}:1:1[/bold]:"),
            call().print(" [bold]This is a long file[/bold]"),
            call().print("[bold]warning:[/bold] this is a long file"),
            call().print(),
        ]

    def test_long_file_delete_fix(self, long_file):
        with patch(
            "sys.argv",
            [
                "check-test",
                "--fix",
                long_file.name,
            ],
        ), self.mock_console() as console, pytest.raises(SystemExit, match=r"^1$"):
            m = LintMain()
            with m.execute() as ctx:
                ctx.add_check(self.long_delete_fix_check)
        assert long_file.read() == "This is a short file now"
        assert console.mock_calls == [
            call(highlight=False),
            call().print(f"In file [bold]{long_file.name}:1:1[/bold]:"),
            call().print(" [bold]This is a long file[/bold]"),
            call().print("[bold]warning:[/bold] this is a long file"),
            call().print(),
            call().print(f"In file [bold]{long_file.name}:1:1[/bold]:"),
            call().print("[red]-[bold]This is a long file[/bold][/red]"),
            call().print("[green]+[bold]This is a short file now[/bold][/green]"),
            call().print(
                "[bold]note:[/bold] suggested fix applied but is too long to display"
            ),
            call().print(),
        ]

    def test_bracket_file(self, bracket_file):
        with patch(
            "sys.argv",
            [
                "check-test",
                "--fix",
                bracket_file.name,
            ],
        ), self.mock_console() as console, pytest.raises(SystemExit, match=r"^1$"):
            m = LintMain()
            with m.execute() as ctx:
                ctx.add_check(self.bracket_check)
        assert bracket_file.read() == "This [file] [has more] [brackets]\n"
        assert console.mock_calls == [
            call(highlight=False),
            call().print(
                rf"In file [bold]{os.path.dirname(bracket_file.name)}"
                r"/file\[with]brackets.txt:1:1[/bold]:"
            ),
            call().print(r" [bold]This \[file] \[has] \[brackets][/bold]"),
            call().print(r"[bold]warning:[/bold] this \[file] has brackets"),
            call().print(),
            call().print(
                rf"In file [bold]{os.path.dirname(bracket_file.name)}"
                r"/file\[with]brackets.txt:1:13[/bold]:"
            ),
            call().print(r"[red]-This \[file] [bold]\[has][/bold] \[brackets][/red]"),
            call().print(
                r"[green]+This \[file] [bold]\[has more][/bold] \[brackets][/green]"
            ),
            call().print("[bold]note:[/bold] suggested fix applied"),
            call().print(),
        ]
