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

import copy
import uuid

import tomlkit
import tomlkit.exceptions

from .lint import LintMain

RAPIDS_LICENSE = "Apache 2.0"
ACCEPTABLE_LICENSES = {
    "Apache 2.0",
    "BSD-3-Clause",
}


def find_value_location(document, key):
    copied_document = copy.deepcopy(document)
    placeholder = uuid.uuid4()

    node = copied_document
    while len(key) > 1:
        node = node[key[0]]
        key = key[1:]
    old_value = node[key[0]]
    node[key[0]] = str(placeholder)

    begin_loc = copied_document.as_string().find(
        tomlkit.string(str(placeholder)).as_string()
    )
    end_loc = begin_loc + len(old_value.as_string())
    return begin_loc, end_loc


def check_pyproject_license(linter, args):
    document = tomlkit.loads(linter.content)
    try:
        project_table = document["project"]
    except tomlkit.exceptions.NonExistentKey:
        loc = (len(linter.content), len(linter.content))
        linter.add_warning(
            loc, f'add project.license with value {{ text = "{RAPIDS_LICENSE}" }}'
        ).add_replacement(
            loc,
            "[project]\nlicense = "
            f"{{ text = {tomlkit.string(RAPIDS_LICENSE).as_string()} }}\n",
        )
        return

    try:
        license_value = project_table["license"]["text"]
    except tomlkit.exceptions.NonExistentKey:
        if project_table.is_super_table():
            loc = (len(linter.content), len(linter.content))
            linter.add_warning(
                loc, f'add project.license with value {{ text = "{RAPIDS_LICENSE}" }}'
            ).add_replacement(
                loc,
                "[project]\nlicense = "
                f"{{ text = {tomlkit.string(RAPIDS_LICENSE).as_string()} }}\n",
            )
        else:
            placeholder = uuid.uuid4()
            copied_document = copy.deepcopy(document)
            copied_document["project"].add(
                str(placeholder), tomlkit.string(str(placeholder))
            )
            index = copied_document.as_string().find(
                f"{placeholder} = {tomlkit.string(str(placeholder)).as_string()}"
            )

            loc = (index, index)
            linter.add_warning(
                loc, f'add project.license with value {{ text = "{RAPIDS_LICENSE}" }}'
            ).add_replacement(
                loc,
                "license = "
                f"{{ text = {tomlkit.string(RAPIDS_LICENSE).as_string()} }}\n",
            )
        return

    if license_value not in ACCEPTABLE_LICENSES:
        loc = find_value_location(document, ("project", "license", "text"))
        linter.add_warning(
            loc, f'license should be "{RAPIDS_LICENSE}"'
        ).add_replacement(loc, tomlkit.string(RAPIDS_LICENSE).as_string())


def main():
    m = LintMain()
    m.argparser.description = (
        f'Verify that pyproject.toml has the correct license ("{RAPIDS_LICENSE}").'
    )
    with m.execute() as ctx:
        ctx.add_check(check_pyproject_license)


if __name__ == "__main__":
    main()
