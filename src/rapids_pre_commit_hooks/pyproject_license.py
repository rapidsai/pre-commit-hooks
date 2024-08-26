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
import copy
import uuid

import tomlkit
import tomlkit.exceptions

from .lint import Linter, LintMain

RAPIDS_LICENSE: str = "Apache 2.0"
ACCEPTABLE_LICENSES: set[str] = {
    RAPIDS_LICENSE,
    "BSD-3-Clause",
}


_LocType = tuple[int, int]


def find_value_location(
    document: tomlkit.TOMLDocument, key: tuple[str, ...], append: bool
) -> _LocType:
    copied_document = copy.deepcopy(document)
    placeholder = uuid.uuid4()
    placeholder_toml = tomlkit.string(str(placeholder))
    placeholder_repr = placeholder_toml.as_string()

    # tomlkit does not provide "mark" information to determine where exactly in the
    # document a value is located, so instead we replace it with a placeholder and
    # look for that in the new document.
    node = copied_document
    while len(key) > (0 if append else 1):
        node = node[key[0]]  # type: ignore[assignment]
        key = key[1:]
    if append:
        node.add(str(placeholder), placeholder_toml)
    else:
        old_value = node[key[0]]
        node[key[0]] = str(placeholder)

    value_to_find = (
        f"{placeholder} = {placeholder_repr}" if append else placeholder_repr
    )
    begin_loc = copied_document.as_string().find(value_to_find)
    end_loc = begin_loc + (0 if append else len(old_value.as_string()))
    return begin_loc, end_loc


def check_pyproject_license(linter: Linter, args: argparse.Namespace) -> None:
    document = tomlkit.loads(linter.content)
    try:
        add_project_table = True
        project_table = document["project"]
        add_project_table = project_table.is_super_table()  # type: ignore[union-attr]
        license_value = project_table["license"]["text"]  # type: ignore[index]
    except tomlkit.exceptions.NonExistentKey:
        if add_project_table:
            loc = (len(linter.content), len(linter.content))
            linter.add_warning(
                loc, f'add project.license with value {{ text = "{RAPIDS_LICENSE}" }}'
            ).add_replacement(
                loc,
                "[project]\nlicense = "
                f"{{ text = {tomlkit.string(RAPIDS_LICENSE).as_string()} }}\n",
            )
        else:
            loc = find_value_location(document, ("project",), True)
            linter.add_warning(
                loc, f'add project.license with value {{ text = "{RAPIDS_LICENSE}" }}'
            ).add_replacement(
                loc,
                "license = "
                f"{{ text = {tomlkit.string(RAPIDS_LICENSE).as_string()} }}\n",
            )
        return

    if license_value not in ACCEPTABLE_LICENSES:
        loc = find_value_location(document, ("project", "license", "text"), False)
        linter.add_warning(loc, f'license should be "{RAPIDS_LICENSE}"')


def main() -> None:
    m = LintMain()
    m.argparser.description = (
        f'Verify that pyproject.toml has the correct license ("{RAPIDS_LICENSE}").'
    )
    with m.execute() as ctx:
        ctx.add_check(check_pyproject_license)


if __name__ == "__main__":
    main()
