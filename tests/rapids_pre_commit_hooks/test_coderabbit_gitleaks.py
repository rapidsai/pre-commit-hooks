# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import json
from unittest.mock import MagicMock, patch

from rapids_pre_commit_hooks.coderabbit_gitleaks import (
    run_coderabbit_gitleaks,
)


@patch("rapids_pre_commit_hooks.coderabbit_gitleaks.subprocess.run")
def test_coderabbit_not_installed(mock_run):
    """Test handling when CodeRabbit CLI is not installed."""
    # Simulate coderabbit command not found
    mock_run.side_effect = FileNotFoundError()

    exit_code = run_coderabbit_gitleaks([])
    assert exit_code == 1


@patch("rapids_pre_commit_hooks.coderabbit_gitleaks.subprocess.run")
def test_coderabbit_version_check_fails(mock_run):
    """Test handling when CodeRabbit version check fails."""
    # Simulate version check returning non-zero
    mock_result = MagicMock()
    mock_result.returncode = 1
    mock_result.stdout = ""
    mock_result.stderr = "command not found"
    mock_run.return_value = mock_result

    exit_code = run_coderabbit_gitleaks([])
    assert exit_code == 1


@patch("rapids_pre_commit_hooks.coderabbit_gitleaks.subprocess.run")
def test_no_secrets_found(mock_run):
    """Test successful scan with no secrets found."""
    # First call: version check (success)
    version_result = MagicMock()
    version_result.returncode = 0
    version_result.stdout = "coderabbit version 1.0.0"

    # Second call: gitleaks scan (no secrets)
    scan_result = MagicMock()
    scan_result.returncode = 0
    scan_result.stdout = "[]"
    scan_result.stderr = ""

    mock_run.side_effect = [version_result, scan_result]

    exit_code = run_coderabbit_gitleaks(["file1.py", "file2.py"])
    assert exit_code == 0


@patch("rapids_pre_commit_hooks.coderabbit_gitleaks.subprocess.run")
def test_secrets_found_with_json_output(mock_run):
    """Test scan with secrets found and valid JSON output."""
    # First call: version check (success)
    version_result = MagicMock()
    version_result.returncode = 0
    version_result.stdout = "coderabbit version 1.0.0"

    # Second call: gitleaks scan (secrets found)
    secrets = [
        {
            "File": "config.py",
            "StartLine": 10,
            "RuleID": "generic-api-key",
            "Match": "api_key = 'sk_test_1234567890'",
        },
        {
            "File": "credentials.json",
            "StartLine": 5,
            "RuleID": "aws-access-token",
            "Match": "AKIAIOSFODNN7EXAMPLE",
        },
    ]
    scan_result = MagicMock()
    scan_result.returncode = 1  # Non-zero indicates secrets found
    scan_result.stdout = json.dumps(secrets)
    scan_result.stderr = ""

    mock_run.side_effect = [version_result, scan_result]

    exit_code = run_coderabbit_gitleaks(["file1.py"])
    assert exit_code == 1


@patch("rapids_pre_commit_hooks.coderabbit_gitleaks.subprocess.run")
def test_secrets_found_no_json_output(mock_run):
    """Test scan with secrets found but no parseable JSON output."""
    # First call: version check (success)
    version_result = MagicMock()
    version_result.returncode = 0
    version_result.stdout = "coderabbit version 1.0.0"

    # Second call: gitleaks scan (secrets found, no JSON)
    scan_result = MagicMock()
    scan_result.returncode = 1  # Non-zero indicates secrets found
    scan_result.stdout = ""
    scan_result.stderr = "Error: secrets detected"

    mock_run.side_effect = [version_result, scan_result]

    exit_code = run_coderabbit_gitleaks([])
    assert exit_code == 1


@patch("rapids_pre_commit_hooks.coderabbit_gitleaks.subprocess.run")
def test_secrets_with_long_match_text(mock_run):
    """Test that long match text is properly truncated."""
    # First call: version check (success)
    version_result = MagicMock()
    version_result.returncode = 0
    version_result.stdout = "coderabbit version 1.0.0"

    # Second call: gitleaks scan with very long match
    long_match = "x" * 100
    secrets = [
        {
            "File": "test.py",
            "StartLine": 1,
            "RuleID": "generic-api-key",
            "Match": long_match,
        }
    ]
    scan_result = MagicMock()
    scan_result.returncode = 1
    scan_result.stdout = json.dumps(secrets)
    scan_result.stderr = ""

    mock_run.side_effect = [version_result, scan_result]

    exit_code = run_coderabbit_gitleaks(["test.py"])
    assert exit_code == 1


@patch("rapids_pre_commit_hooks.coderabbit_gitleaks.subprocess.run")
def test_gitleaks_scan_exception(mock_run):
    """Test handling of exceptions during gitleaks scan."""
    # First call: version check (success)
    version_result = MagicMock()
    version_result.returncode = 0
    version_result.stdout = "coderabbit version 1.0.0"

    # Second call: gitleaks scan raises exception
    mock_run.side_effect = [
        version_result,
        Exception("Unexpected error"),
    ]

    exit_code = run_coderabbit_gitleaks([])
    assert exit_code == 1


@patch("rapids_pre_commit_hooks.coderabbit_gitleaks.subprocess.run")
def test_invalid_json_output(mock_run):
    """Test handling of invalid JSON output from gitleaks."""
    # First call: version check (success)
    version_result = MagicMock()
    version_result.returncode = 0
    version_result.stdout = "coderabbit version 1.0.0"

    # Second call: gitleaks scan with invalid JSON
    scan_result = MagicMock()
    scan_result.returncode = 1
    scan_result.stdout = "not valid json {["
    scan_result.stderr = "some error occurred"

    mock_run.side_effect = [version_result, scan_result]

    exit_code = run_coderabbit_gitleaks([])
    assert exit_code == 1


@patch("rapids_pre_commit_hooks.coderabbit_gitleaks.subprocess.run")
def test_empty_secrets_list(mock_run):
    """Test scan with empty secrets list (no secrets)."""
    # First call: version check (success)
    version_result = MagicMock()
    version_result.returncode = 0
    version_result.stdout = "coderabbit version 1.0.0"

    # Second call: gitleaks scan (empty list)
    scan_result = MagicMock()
    scan_result.returncode = 0
    scan_result.stdout = "[]"
    scan_result.stderr = ""

    mock_run.side_effect = [version_result, scan_result]

    exit_code = run_coderabbit_gitleaks(["file1.py"])
    assert exit_code == 0


@patch("rapids_pre_commit_hooks.coderabbit_gitleaks.subprocess.run")
def test_secrets_with_minimal_fields(mock_run):
    """Test secrets output with minimal required fields."""
    # First call: version check (success)
    version_result = MagicMock()
    version_result.returncode = 0
    version_result.stdout = "coderabbit version 1.0.0"

    # Second call: gitleaks scan with minimal fields
    secrets = [
        {
            "File": "secret.txt",
            # Missing StartLine and RuleID
        }
    ]
    scan_result = MagicMock()
    scan_result.returncode = 1
    scan_result.stdout = json.dumps(secrets)
    scan_result.stderr = ""

    mock_run.side_effect = [version_result, scan_result]

    exit_code = run_coderabbit_gitleaks(["secret.txt"])
    assert exit_code == 1


@patch("rapids_pre_commit_hooks.coderabbit_gitleaks.subprocess.run")
def test_multiple_secrets_in_same_file(mock_run):
    """Test multiple secrets detected in the same file."""
    # First call: version check (success)
    version_result = MagicMock()
    version_result.returncode = 0
    version_result.stdout = "coderabbit version 1.0.0"

    # Second call: gitleaks scan with multiple secrets
    secrets = [
        {
            "File": "config.py",
            "StartLine": 10,
            "RuleID": "generic-api-key",
            "Match": "api_key = 'secret1'",
        },
        {
            "File": "config.py",
            "StartLine": 20,
            "RuleID": "generic-secret",
            "Match": "password = 'secret2'",
        },
        {
            "File": "other.py",
            "StartLine": 5,
            "RuleID": "aws-access-token",
            "Match": "aws_token",
        },
    ]
    scan_result = MagicMock()
    scan_result.returncode = 1
    scan_result.stdout = json.dumps(secrets)
    scan_result.stderr = ""

    mock_run.side_effect = [version_result, scan_result]

    exit_code = run_coderabbit_gitleaks(["config.py", "other.py"])
    assert exit_code == 1


@patch.dict(
    "os.environ", {"SKIP_CODERABBIT_IF_MISSING": "true"}, clear=False
)
@patch("rapids_pre_commit_hooks.coderabbit_gitleaks.subprocess.run")
def test_skip_when_not_installed_and_env_var_set(mock_run):
    """Test that hook passes with warning when skip flag is set."""
    # Simulate coderabbit command not found
    mock_run.side_effect = FileNotFoundError()

    exit_code = run_coderabbit_gitleaks([])
    assert exit_code == 0


@patch.dict(
    "os.environ", {"SKIP_CODERABBIT_IF_MISSING": "1"}, clear=False
)
@patch("rapids_pre_commit_hooks.coderabbit_gitleaks.subprocess.run")
def test_skip_with_numeric_env_var(mock_run):
    """Test that hook passes with warning when skip flag is '1'."""
    # Simulate coderabbit command not found
    mock_run.side_effect = FileNotFoundError()

    exit_code = run_coderabbit_gitleaks([])
    assert exit_code == 0


@patch.dict(
    "os.environ", {"SKIP_CODERABBIT_IF_MISSING": "false"}, clear=False
)
@patch("rapids_pre_commit_hooks.coderabbit_gitleaks.subprocess.run")
def test_no_skip_when_env_var_is_false(mock_run):
    """Test that hook fails when skip flag is explicitly false."""
    # Simulate coderabbit command not found
    mock_run.side_effect = FileNotFoundError()

    exit_code = run_coderabbit_gitleaks([])
    assert exit_code == 1


@patch("rapids_pre_commit_hooks.coderabbit_gitleaks.subprocess.run")
def test_no_files_passed(mock_run):
    """Test running scan with no files (scans entire repository)."""
    # First call: version check (success)
    version_result = MagicMock()
    version_result.returncode = 0
    version_result.stdout = "coderabbit version 1.0.0"

    # Second call: gitleaks scan (no secrets)
    scan_result = MagicMock()
    scan_result.returncode = 0
    scan_result.stdout = "[]"
    scan_result.stderr = ""

    mock_run.side_effect = [version_result, scan_result]

    exit_code = run_coderabbit_gitleaks([])
    assert exit_code == 0

    # Verify that gitleaks was called without file arguments
    assert mock_run.call_count == 2
    gitleaks_call = mock_run.call_args_list[1]
    assert gitleaks_call[0][0] == [
        "coderabbit",
        "gitleaks",
        "--format",
        "json",
    ]
