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

import re

from packaging.version import Version

from .lint import LintMain


def check_rapids_version(linter, rapids_version_re):
    for match in rapids_version_re.finditer(linter.content):
        linter.add_warning(
            match.span(),
            "do not hard-code RAPIDS version; dynamically read from VERSION file or "
            f'write a "{linter.filename}.rapids_metadata_template" file',
        )


def check_rapids_metadata():
    with open("VERSION") as f:
        version = Version(f.read())

    rapids_version_re = re.compile(
        rf"{version.major:02}\.{version.minor:02}(\.{version.micro:02})?"
    )

    def the_check(linter, args):
        try:
            with open(f"{linter.filename}.rapids_metadata_template") as f:
                template_content = f.read()
        except FileNotFoundError:
            template_content = None

        if template_content is None:
            check_rapids_version(linter, rapids_version_re)
        else:
            template_replacement = template_content.format(
                RAPIDS_VERSION_MAJOR=f"{version.major:02}",
                RAPIDS_VERSION_MINOR=f"{version.minor:02}",
                RAPIDS_VERSION_PATCH=f"{version.micro:02}",
                RAPIDS_VERSION=(
                    f"{version.major:02}.{version.minor:02}.{version.micro:02}"
                ),
                RAPIDS_VERSION_MAJOR_MINOR=f"{version.major:02}.{version.minor:02}",
            )

            if linter.content != template_replacement:
                linter.add_warning(
                    (0, len(linter.content)),
                    f'file does not match template replacement from "{linter.filename}'
                    '.rapids_metadata_template"',
                ).add_replacement((0, len(linter.content)), template_replacement)

    return the_check


def main():
    m = LintMain()
    with m.execute() as ctx:
        ctx.add_check(check_rapids_metadata())


if __name__ == "__main__":
    main()
