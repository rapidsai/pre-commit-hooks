# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import os
import subprocess
import sys
from typing import TYPE_CHECKING

from rich.console import Console
from rich.markup import escape

if TYPE_CHECKING:
    from typing import Any


def run_coderabbit_gitleaks(files: list[str]) -> int:
    """Run CodeRabbit CLI Gitleaks scan and check for secrets.

    Args:
        files: List of files to scan (passed from pre-commit)

    Returns:
        Exit code: 0 if no secrets found, 1 if secrets found or error occurred
    """
    console = Console(highlight=False)

    # Allow skipping if CodeRabbit is not installed (for testing/CI)
    skip_if_missing = os.getenv("SKIP_CODERABBIT_IF_MISSING", "false").lower() in (
        "true",
        "1",
        "yes",
    )

    # Check if coderabbit CLI is installed
    try:
        result = subprocess.run(
            ["coderabbit", "--version"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            if skip_if_missing:
                console.print(
                    "[bold yellow]Warning:[/bold yellow] CodeRabbit CLI is "
                    "not installed - skipping secret scan"
                )
                return 0
            console.print(
                "[bold red]Error:[/bold red] CodeRabbit CLI is not installed "
                "or not in PATH"
            )
            console.print(
                "Install it from: https://docs.coderabbit.ai/cli#installation"
            )
            return 1
    except FileNotFoundError:
        if skip_if_missing:
            console.print(
                "[bold yellow]Warning:[/bold yellow] CodeRabbit CLI is not "
                "installed - skipping secret scan"
            )
            return 0
        console.print(
            "[bold red]Error:[/bold red] CodeRabbit CLI is not installed"
        )
        console.print(
            "Install it from: https://docs.coderabbit.ai/cli#installation"
        )
        return 1

    # Run gitleaks scan
    console.print("Running CodeRabbit Gitleaks scan...")
    try:
        result = subprocess.run(
            ["coderabbit", "gitleaks", "--format", "json"],
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception as e:
        console.print(f"[bold red]Error running CodeRabbit:[/bold red] {e}")
        return 1

    # Parse the output
    has_secrets = False
    if result.returncode != 0:
        # Gitleaks returns non-zero when secrets are found
        has_secrets = True

    # Try to parse JSON output for detailed information
    if result.stdout:
        try:
            output_data: "Any" = json.loads(result.stdout)
            if isinstance(output_data, list) and len(output_data) > 0:
                has_secrets = True
                console.print(
                    f"\n[bold red]Found {len(output_data)} potential "
                    f"secret(s):[/bold red]\n"
                )
                for i, finding in enumerate(output_data, 1):
                    file_path = finding.get("File", "unknown")
                    line = finding.get("StartLine", "?")
                    rule_id = finding.get("RuleID", "unknown")
                    match_text = finding.get("Match", "")

                    console.print(
                        f"{i}. [bold]{escape(file_path)}:{line}[/bold]"
                    )
                    console.print(f"   Rule: {escape(rule_id)}")
                    if match_text:
                        # Truncate long matches
                        if len(match_text) > 80:
                            match_text = match_text[:77] + "..."
                        console.print(f"   Match: {escape(match_text)}")
                    console.print()
        except json.JSONDecodeError:
            # If JSON parsing fails, check stderr for error messages
            if result.stderr:
                console.print(
                    "[bold yellow]Warning:[/bold yellow] "
                    "Could not parse Gitleaks output"
                )
                console.print(result.stderr)

    if has_secrets:
        console.print(
            "[bold red]CodeRabbit Gitleaks found potential secrets in your "
            "code.[/bold red]"
        )
        console.print(
            "Please review and remove any sensitive information before "
            "committing."
        )
        return 1

    console.print("[bold green]No secrets detected.[/bold green]")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run CodeRabbit CLI Gitleaks scan to detect secrets"
    )
    parser.add_argument(
        "files",
        nargs="*",
        metavar="file",
        help="files to check (passed by pre-commit)",
    )
    args = parser.parse_args()

    # Run the scan - CodeRabbit Gitleaks scans the entire repository
    # regardless of which files are passed
    sys.exit(run_coderabbit_gitleaks(args.files))


if __name__ == "__main__":
    main()
