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
import random

import tomlkit
import tomlkit.exceptions

from .lint import LintMain

RAPIDS_LICENSE = "Apache 2.0"


def find_value_location(document, key):
    copied_document = copy.deepcopy(document)
    node = copied_document
    while str(placeholder := random.randint(0, 1048576)) in node.as_string():
        pass

    while len(key) > 1:
        node = node[key[0]]
        key = key[1:]
    node[key[0]] = placeholder

    begin_loc = copied_document.as_string().find(str(placeholder))
    end_loc = (
        begin_loc
        + len(str(placeholder))
        - len(copied_document.as_string())
        + len(document.as_string())
    )
    return begin_loc, end_loc


def check_pyproject_license(linter, args):
    document = tomlkit.loads(linter.content)
    try:
        license_value = document["project"]["license"]["text"]
    except tomlkit.exceptions.NonExistentKey:
        return

    if license_value != RAPIDS_LICENSE:
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
