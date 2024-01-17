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

import bashlex

from ..lint import LintMain


class LintVisitor(bashlex.ast.nodevisitor):
    def __init__(self, linter, args):
        self.linter = linter
        self.args = args

    def add_warning(self, pos, msg):
        return self.linter.add_warning(pos, msg)


class ShellMain(LintMain):
    def __init__(self):
        super().__init__()
        self.visitors = []
        self.add_check(self.check_shell)

    def add_visitor_class(self, cls):
        self.visitors.append(cls)

    def check_shell(self, linter, args):
        parts = bashlex.parse(linter.content)

        for cls in self.visitors:
            visitor = cls(linter, args)
            for part in parts:
                visitor.visit(part)
