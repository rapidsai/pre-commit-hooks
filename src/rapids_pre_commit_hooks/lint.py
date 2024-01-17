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

import argparse
import bisect
import contextlib
import functools
import itertools


class OverlappingReplacementsError(RuntimeError):
    pass


class Replacement:
    def __init__(self, pos, newtext):
        self.pos = pos
        self.newtext = newtext

    def __eq__(self, other):
        if not isinstance(other, Replacement):
            return False
        return self.pos == other.pos and self.newtext == other.newtext

    def __repr__(self):
        return f"Replacement(pos={self.pos}, newtext={repr(self.newtext)})"


class LintWarning:
    def __init__(self, pos, msg):
        self.pos = pos
        self.msg = msg
        self.replacements = []

    def add_replacement(self, pos, newtext):
        self.replacements.append(Replacement(pos, newtext))

    def __eq__(self, other):
        if not isinstance(other, LintWarning):
            return False
        return (
            self.pos == other.pos
            and self.msg == other.msg
            and self.replacements == other.replacements
        )

    def __repr__(self):
        return (
            "LintWarning("
            + f"pos={self.pos}, "
            + f"msg={self.msg}, "
            + f"replacements={self.replacements})"
        )


class Linter:
    def __init__(self, filename, content):
        self.filename = filename
        self.content = content
        self.warnings = []
        self._calculate_lines()

    def add_warning(self, pos, msg):
        w = LintWarning(pos, msg)
        self.warnings.append(w)
        return w

    def fix(self):
        sorted_replacements = sorted(
            (
                replacement
                for warning in self.warnings
                for replacement in warning.replacements
            ),
            key=lambda replacement: replacement.pos,
        )

        for r1, r2 in itertools.pairwise(sorted_replacements):
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

    def print_warnings(self, fix_applied=False):
        sorted_warnings = sorted(self.warnings, key=lambda warning: warning.pos)

        for warning in sorted_warnings:
            line_index = self.line_for_pos(warning.pos[0])
            print(f"In file {self.filename}:{line_index + 1}:")
            self.print_highlighted_code(warning.pos)
            print(f"warning: {warning.msg}")
            print()

            for replacement in warning.replacements:
                line_index = self.line_for_pos(replacement.pos[0])
                print(f"In file {self.filename}:{line_index + 1}:")
                self.print_highlighted_code(replacement.pos, replacement.newtext)
                if fix_applied:
                    print("note: suggested fix applied")
                else:
                    print("note: suggested fix")
                print()

    def print_highlighted_code(self, pos, replacement=""):
        line_index = self.line_for_pos(pos[0])
        line_pos = self.lines[line_index]
        left = pos[0] - line_pos[0]

        if self.line_for_pos(pos[1]) == line_index:
            right = pos[1] - line_pos[0]
        else:
            right = line_pos[1] - line_pos[0]
        length = right - left

        print(self.content[line_pos[0] : line_pos[1]])
        print(" " * left, end="")
        if length == 0:
            print(f"^{replacement}")
        else:
            print(f"{'~' * length}{replacement}")

    def line_for_pos(self, index):
        @functools.total_ordering
        class LineComparator:
            def __init__(self, pos):
                self.pos = pos

            def __lt__(self, other):
                return self.pos[1] < other

            def __gt__(self, other):
                return self.pos[0] > other

            def __eq__(self, other):
                return self.pos[0] <= other <= self.pos[1]

        line_index = bisect.bisect_left(self.lines, index, key=LineComparator)
        try:
            line_pos = self.lines[line_index]
        except IndexError:
            return None
        if line_pos[0] <= index <= line_pos[1]:
            return line_index
        return None

    def _calculate_lines(self):
        self.lines = []

        line_begin = 0
        line_end = 0
        state = "c"

        for c in self.content:
            if state == "c":
                if c == "\r":
                    self.lines.append((line_begin, line_end))
                    line_end = line_begin = line_end + 1
                    state = "r"
                elif c == "\n":
                    self.lines.append((line_begin, line_end))
                    line_end = line_begin = line_end + 1
                else:
                    line_end += 1
            elif state == "r":
                if c == "\r":
                    self.lines.append((line_begin, line_end))
                    line_end = line_begin = line_end + 1
                elif c == "\n":
                    line_end = line_begin = line_end + 1
                    state = "c"
                else:
                    line_end += 1
                    state = "c"

        self.lines.append((line_begin, line_end))


class LintMain(contextlib.AbstractContextManager):
    def __init__(self):
        self.argparser = argparse.ArgumentParser()
        self.argparser.add_argument("--fix", action="store_true")
        self.argparser.add_argument("file", nargs="+")
        self.checks = []

    def add_check(self, check):
        self.checks.append(check)

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type:
            return

        warnings = False

        args = self.argparser.parse_args()

        for file in args.file:
            with open(file) as f:
                content = f.read()

            linter = Linter(file, content)
            for check in self.checks:
                check(linter, args)

            linter.print_warnings(args.fix)
            if args.fix:
                fix = linter.fix()
                if fix != content:
                    with open(file, "w") as f:
                        f.write(fix)

            if len(linter.warnings) > 0:
                warnings = True

        if warnings:
            exit(1)
