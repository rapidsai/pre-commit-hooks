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
    document: "tomlkit.TOMLDocument",
    key: tuple[str, ...],
    *,
    append: bool,
) -> _LocType:
    """
    Find the exact location of a key in a stringified TOML document.

    Parameters
    ----------
    document : tomlkit.TOMLDocument
        TOML content
    key : tuple[str, ...]
        Tuple of strings, of any length.
        Items are evaluated in order as keys to subset into ``document``.
        For example, to reference the 'license' value in the [project] table
        in a pyproject.toml, ``key = ("project", "license",)``.
    append : bool
        If ``True``, returns the location where new text will be added.
        If ``False``, returns the range of characters to be replaced.

    Returns
    -------
    loc : tuple[int, int]
        Location of the key and its value in the document.
        e.g., ``(20, 35)`` = "the 20th-35th characters, including newlines"
          * element 0: number of characters from beginning of the document to
                       beginning of the section indicated by ``key``
          * element 1: final character to replace
    """
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
        value_to_find = f"{placeholder} = {placeholder_repr}"
        begin_loc = copied_document.as_string().find(value_to_find)
        return begin_loc, begin_loc

    # otherwise, if replacing without appending
    old_value = node[key[0]]
    node[key[0]] = str(placeholder)
    begin_loc = copied_document.as_string().find(placeholder_repr)
    end_loc = begin_loc + len(old_value.as_string())
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
            loc = find_value_location(document, ("project",), append=True)
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
        loc = find_value_location(
            document, ("project", "license"), append=False
        )
        if license_value := document["project"]["license"].get("text", None):  # type: ignore[index, union-attr]
            slugified_license_value = re.sub(
                r"\s+", "-", str(license_value).strip()
            )
            if slugified_license_value in ACCEPTABLE_LICENSES:
                linter.add_warning(
                    loc, f'license should be "{slugified_license_value}"'
                ).add_replacement(
                    loc,
                    "license = "
                    + f"{tomlkit.string(slugified_license_value).as_string()}",
                )
            else:
                linter.add_warning(
                    loc,
                    f'license should be "{RAPIDS_LICENSE}"'
                    + f', got {{ license = {{ text = "{license_value}" }} }}',
                )
        else:
            linter.add_warning(loc, f'license should be "{RAPIDS_LICENSE}"')
        return

    if license_value not in ACCEPTABLE_LICENSES:
        loc = find_value_location(
            document, ("project", "license"), append=False
        )
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
                + f"{tomlkit.string(slugified_license_value).as_string()}",
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
