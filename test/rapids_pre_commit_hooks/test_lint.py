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
import sys
import tempfile

import pytest

from rapids_pre_commit_hooks.lint import Linter, LintMain, OverlappingReplacementsError


class MockArgv(contextlib.AbstractContextManager):
    def __init__(self, *argv):
        self.argv = argv

    def __enter__(self):
        self.old_argv = sys.argv
        sys.argv = list(self.argv)

    def __exit__(self, exc_type, exc_value, traceback):
        sys.argv = self.old_argv


class TestLinter:
    def test_lines(self):
        linter = Linter(
            "test.txt",
            "line 1\nline 2\rline 3\r\nline 4\r\n\nline 6\r\n\r\nline 8\n\r\n"
            + "line 10\r\r\nline 12\r\n\rline 14\n\nline 16\r\rline 18\n\rline 20",
        )
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

    def test_line_for_pos(self):
        linter = Linter(
            "test.txt",
            "line 1\nline 2\rline 3\r\nline 4\r\n\nline 6\r\n\r\nline 8\n\r\n"
            + "line 10\r\r\nline 12\r\n\rline 14\n\nline 16\r\rline 18\n\rline 20",
        )
        assert linter.line_for_pos(0) == 0
        assert linter.line_for_pos(3) == 0
        assert linter.line_for_pos(6) == 0
        assert linter.line_for_pos(10) == 1
        assert linter.line_for_pos(21) is None
        assert linter.line_for_pos(34) == 5
        assert linter.line_for_pos(97) == 19
        assert linter.line_for_pos(104) == 19
        assert linter.line_for_pos(200) is None

        linter = Linter("test.txt", "line 1")
        assert linter.line_for_pos(0) == 0
        assert linter.line_for_pos(3) == 0
        assert linter.line_for_pos(6) == 0

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
    def hello_world_file(self):
        with tempfile.NamedTemporaryFile("w+") as f:
            f.write("Hello world!")
            f.flush()
            f.seek(0)
            yield f

    @pytest.fixture
    def hello_file(self):
        with tempfile.NamedTemporaryFile("w+") as f:
            f.write("Hello!")
            f.flush()
            f.seek(0)
            yield f

    def the_check(self, linter, args):
        assert args.check_test
        linter.add_warning((0, 5), "say good bye instead").add_replacement(
            (0, 5), "Good bye"
        )
        if linter.content[5] != "!":
            linter.add_warning((5, 5), "use punctuation").add_replacement((5, 5), ",")

    def test_no_warnings_no_fix(self, hello_world_file, capsys):
        with MockArgv("check-test", "--check-test", hello_world_file.name):
            with LintMain() as m:
                m.argparser.add_argument("--check-test", action="store_true")
        assert hello_world_file.read() == "Hello world!"
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_no_warnings_fix(self, hello_world_file, capsys):
        with MockArgv("check-test", "--check-test", "--fix", hello_world_file.name):
            with LintMain() as m:
                m.argparser.add_argument("--check-test", action="store_true")
        assert hello_world_file.read() == "Hello world!"
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_warnings_no_fix(self, hello_world_file, capsys):
        with MockArgv(
            "check-test", "--check-test", hello_world_file.name
        ), pytest.raises(SystemExit, match=r"^1$"):
            with LintMain() as m:
                m.argparser.add_argument("--check-test", action="store_true")
                m.add_check(self.the_check)
        assert hello_world_file.read() == "Hello world!"
        captured = capsys.readouterr()
        assert (
            captured.out
            == f"""In file {hello_world_file.name}:1:
Hello world!
~~~~~
warning: say good bye instead

In file {hello_world_file.name}:1:
Hello world!
~~~~~Good bye
note: suggested fix

In file {hello_world_file.name}:1:
Hello world!
     ^
warning: use punctuation

In file {hello_world_file.name}:1:
Hello world!
     ^,
note: suggested fix

"""
        )

    def test_warnings_fix(self, hello_world_file, capsys):
        with MockArgv(
            "check-test", "--check-test", "--fix", hello_world_file.name
        ), pytest.raises(SystemExit, match=r"^1$"):
            with LintMain() as m:
                m.argparser.add_argument("--check-test", action="store_true")
                m.add_check(self.the_check)
        assert hello_world_file.read() == "Good bye, world!"
        captured = capsys.readouterr()
        assert (
            captured.out
            == f"""In file {hello_world_file.name}:1:
Hello world!
~~~~~
warning: say good bye instead

In file {hello_world_file.name}:1:
Hello world!
~~~~~Good bye
note: suggested fix applied

In file {hello_world_file.name}:1:
Hello world!
     ^
warning: use punctuation

In file {hello_world_file.name}:1:
Hello world!
     ^,
note: suggested fix applied

"""
        )

    def test_multiple_files(self, hello_world_file, hello_file, capsys):
        with MockArgv(
            "check-test",
            "--check-test",
            "--fix",
            hello_world_file.name,
            hello_file.name,
        ), pytest.raises(SystemExit, match=r"^1$"):
            with LintMain() as m:
                m.argparser.add_argument("--check-test", action="store_true")
                m.add_check(self.the_check)
        assert hello_world_file.read() == "Good bye, world!"
        assert hello_file.read() == "Good bye!"
        captured = capsys.readouterr()
        assert (
            captured.out
            == f"""In file {hello_world_file.name}:1:
Hello world!
~~~~~
warning: say good bye instead

In file {hello_world_file.name}:1:
Hello world!
~~~~~Good bye
note: suggested fix applied

In file {hello_world_file.name}:1:
Hello world!
     ^
warning: use punctuation

In file {hello_world_file.name}:1:
Hello world!
     ^,
note: suggested fix applied

In file {hello_file.name}:1:
Hello!
~~~~~
warning: say good bye instead

In file {hello_file.name}:1:
Hello!
~~~~~Good bye
note: suggested fix applied

"""
        )
