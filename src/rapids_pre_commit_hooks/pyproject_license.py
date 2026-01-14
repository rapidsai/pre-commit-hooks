# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import argparse
import copy
import uuid
import re

import tomlkit
import tomlkit.exceptions

from .lint import Linter, LintMain

RAPIDS_LICENSE: str = "Apache-2.0"
ACCEPTABLE_LICENSES: set[str] = {
    RAPIDS_LICENSE,
    "BSD-3-Clause",
}


_LocType = tuple[int, int]


def find_value_location(
    document: "tomlkit.TOMLDocument", key: tuple[str, ...], append: bool
) -> _LocType:
    copied_document = copy.deepcopy(document)
    placeholder = uuid.uuid4()
    placeholder_toml = tomlkit.string(str(placeholder))
    placeholder_repr = placeholder_toml.as_string()

    # tomlkit does not provide "mark" information to determine where exactly in
    # the document a value is located, so instead we replace it with a
    # placeholder and look for that in the new document.
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


def check_pyproject_license(linter: Linter, _args: argparse.Namespace) -> None:
    document = tomlkit.loads(linter.content)
    try:
        add_project_table = True
        project_table = document["project"]
        add_project_table = project_table.is_super_table()  # type: ignore[union-attr]
        license_value = project_table["license"]  # type: ignore[index]
    except tomlkit.exceptions.NonExistentKey:
        if add_project_table:
            loc = (len(linter.content), len(linter.content))
            linter.add_warning(
                loc, f'add project.license with value "{RAPIDS_LICENSE}"'
            ).add_replacement(
                loc,
                f"[project]{linter.lines.newline_style}license = "
                f"{tomlkit.string(RAPIDS_LICENSE).as_string()}"
                + linter.lines.newline_style,
            )
        else:
            loc = find_value_location(document, ("project",), True)
            linter.add_warning(
                loc, f'add project.license with value "{RAPIDS_LICENSE}"'
            ).add_replacement(
                loc,
                f"license = {tomlkit.string(RAPIDS_LICENSE).as_string()}"
                + linter.lines.newline_style,
            )
        return

    # handle case where the license is still in
    # "license = { text = 'something' }" form
    if isinstance(license_value, tomlkit.items.InlineTable):
        loc = find_value_location(document, ("project", "license"), False)
        linter.add_warning(loc, f'license should be "{RAPIDS_LICENSE}"')
        return

    if license_value not in ACCEPTABLE_LICENSES:
        loc = find_value_location(document, ("project", "license"), False)
        slugified_license_value = re.sub(
            r"\s+", "-", str(license_value).strip()
        )
        if slugified_license_value in ACCEPTABLE_LICENSES:
            linter.add_warning(
                loc,
                f'license should be "{slugified_license_value}"'
                + f', got "{license_value}"',
            ).add_replacement(
                loc,
                "license = "
                + f"{tomlkit.string(slugified_license_value).as_string()}"
                + linter.lines.newline_style,
            )
            return

        linter.add_warning(loc, f'license should be "{RAPIDS_LICENSE}"')


def main() -> None:
    m = LintMain("verify-pyproject-license")
    m.argparser.description = "Verify that pyproject.toml has the correct "
    f'license ("{RAPIDS_LICENSE}").'
    with m.execute() as ctx:
        ctx.add_check(check_pyproject_license)


if __name__ == "__main__":
    main()
