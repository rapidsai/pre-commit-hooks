# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import argparse

import bashlex

from ..lint import ExecutionContext, Linter, LintMain, LintWarning

_PosType = tuple[int, int]


class LintVisitor(bashlex.ast.nodevisitor):
    def __init__(self, linter: Linter, args: argparse.Namespace) -> None:
        self.linter: Linter = linter
        self.args: argparse.Namespace = args

    def add_warning(self, pos: _PosType, msg: str) -> LintWarning:
        return self.linter.add_warning(pos, msg)


class ShellExecutionContext(ExecutionContext):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__(args)
        self.visitors: list[type] = []
        self.add_check(self.check_shell)

    def add_visitor_class(self, cls: type) -> None:
        self.visitors.append(cls)

    def check_shell(self, linter: Linter, args: argparse.Namespace) -> None:
        parts = bashlex.parse(linter.content)

        for cls in self.visitors:
            visitor = cls(linter, args)
            for part in parts:
                visitor.visit(part)


class ShellMain(LintMain):
    context_class = ShellExecutionContext
