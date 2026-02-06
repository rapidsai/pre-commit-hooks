# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import argparse
import re

import tomlkit
import tomlkit.exceptions

from .lint import Linter, LintMain
from .utils.toml import find_value_span

RAPIDS_LICENSE: str = "Apache-2.0"
ACCEPTABLE_LICENSES: set[str] = {
    RAPIDS_LICENSE,
    "BSD-3-Clause",
}


def check_pyproject_license(linter: Linter, _args: argparse.Namespace) -> None:
    document = tomlkit.loads(linter.content)
    try:
        # by default, assume the linter needs to add the [project] table
        add_project_table = True
        project_table = document["project"]

        # If there are other non-[project*] tables between the first [project*]
        # and the last one, tomlkit will return an OutOfOrderTableProxy object.
        #
        # Assume this order was unintentional (there's no functional reason to
        # interleave tables like this) and just emit a warning saying this
        # should be fixed.
        #
        # This is easier than having to special-case OutOfOrderTableProxy in
        # the replacement / appending code, and also enforces a bit more
        # standardization in pyproject.toml files (a good thing on its own!).
        if isinstance(project_table, tomlkit.container.OutOfOrderTableProxy):
            span = (len(linter.content), len(linter.content))
            linter.add_warning(
                span,
                (
                    "[project] table should precede all other [project.*] "
                    "tables and all [project.*] tables should be grouped "
                    "together"
                ),
            )
            return

        add_project_table = project_table.is_super_table()  # type: ignore[union-attr]
        license_value = project_table["license"]  # type: ignore[index]
    except tomlkit.exceptions.NonExistentKey:
        if add_project_table:
            span = (len(linter.content), len(linter.content))
            linter.add_warning(
                span, f'add project.license with value "{RAPIDS_LICENSE}"'
            ).add_replacement(
                span,
                f"[project]{linter.lines.newline_style}license = "
                f"{tomlkit.string(RAPIDS_LICENSE).as_string()}"
                + linter.lines.newline_style,
            )
        else:
            span = find_value_span(document, ("project",), append=True)
            linter.add_warning(
                span, f'add project.license with value "{RAPIDS_LICENSE}"'
            ).add_replacement(
                span,
                f"license = {tomlkit.string(RAPIDS_LICENSE).as_string()}"
                + linter.lines.newline_style,
            )
        return

    # handle case where the license is still in
    # "license = { text = 'something' }" form
    if isinstance(license_value, tomlkit.items.InlineTable):
        span = find_value_span(document, ("project", "license"), append=False)
        if license_value := document["project"]["license"].get("text", None):  # type: ignore[index, union-attr]
            slugified_license_value = re.sub(
                r"\s+", "-", str(license_value).strip()
            )
            if slugified_license_value in ACCEPTABLE_LICENSES:
                linter.add_warning(
                    span, f'license should be "{slugified_license_value}"'
                ).add_replacement(
                    span,
                    f"{tomlkit.string(slugified_license_value).as_string()}",
                )
            else:
                linter.add_warning(
                    span,
                    f'license should be "{RAPIDS_LICENSE}"'
                    + f', got license = {{ text = "{license_value}" }}',
                )
        else:
            linter.add_warning(span, f'license should be "{RAPIDS_LICENSE}"')
        return

    if license_value not in ACCEPTABLE_LICENSES:
        span = find_value_span(document, ("project", "license"), append=False)
        slugified_license_value = re.sub(
            r"\s+", "-", str(license_value).strip()
        )
        if slugified_license_value in ACCEPTABLE_LICENSES:
            linter.add_warning(
                span,
                f'license should be "{slugified_license_value}"'
                + f', got "{license_value}"',
            ).add_replacement(
                span,
                f"{tomlkit.string(slugified_license_value).as_string()}",
            )
            return

        linter.add_warning(span, f'license should be "{RAPIDS_LICENSE}"')


def main() -> None:
    m = LintMain("verify-pyproject-license")
    m.argparser.description = "Verify that pyproject.toml has the correct "
    f'license ("{RAPIDS_LICENSE}").'
    with m.execute() as ctx:
        ctx.add_check(check_pyproject_license)


if __name__ == "__main__":
    main()
