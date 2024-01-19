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
    def visitcommand(self, n, parts):
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
        if any(arg in {"-h", "--help", "-V"} for arg in part_words[1:command_index]):
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


def main():
    m = ShellMain()
    with m.execute() as ctx:
        ctx.add_visitor_class(VerifyCondaYesVisitor)


if __name__ == "__main__":
    main()
