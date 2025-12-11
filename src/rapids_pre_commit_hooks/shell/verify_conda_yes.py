# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from . import LintVisitor, ShellMain

INTERACTIVE_CONDA_COMMANDS = {
    "clean": {
        "args": ["-y", "--yes"],
    },
    "create": {
        "args": ["-y", "--yes"],
    },
    "install": {
        "args": ["-y", "--yes"],
    },
    "remove": {
        "args": ["-y", "--yes"],
    },
    "uninstall": {
        "args": ["-y", "--yes"],
    },
    "update": {
        "args": ["-y", "--yes"],
    },
    "upgrade": {
        "args": ["-y", "--yes"],
    },
}


class VerifyCondaYesVisitor(LintVisitor):
    def visitcommand(self, _n, parts) -> None:
        part_words = [part.word for part in parts]
        if part_words[0] != "conda":
            return

        try:
            command_index = next(
                i
                for i, word in enumerate(part_words)
                if word not in {"conda", "-h", "--help", "--no-plugins", "-V"}
            )
        except StopIteration:
            return
        if any(
            arg in {"-h", "--help", "-V"}
            for arg in part_words[1:command_index]
        ):
            return

        command_name = part_words[command_index]
        command_args = part_words[command_index:]
        try:
            command = INTERACTIVE_CONDA_COMMANDS[command_name]
        except KeyError:
            return

        if not any(arg in command["args"] for arg in command_args):
            warning_pos = (parts[0].pos[0], parts[command_index].pos[1])
            insert_pos = (warning_pos[1], warning_pos[1])

            warning = self.add_warning(
                warning_pos, f"add {command['args'][0]} argument"
            )
            warning.add_replacement(insert_pos, f" {command['args'][0]}")


def main() -> None:
    m = ShellMain("verify-conda-yes")
    with m.execute() as ctx:
        ctx.add_visitor_class(VerifyCondaYesVisitor)


if __name__ == "__main__":
    main()
