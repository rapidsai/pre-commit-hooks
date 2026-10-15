---
name: add-or-modify-hook
description: >
  Add a new hook to rapids-pre-commit-hooks, or modify an existing hook
---
# Overview
The hooks in this repository are meant for enforcing RAPIDS configuration and
code conventions. They are intended to be run by
[pre-commit](https://github.com/pre-commit/pre-commit).

# Hooks
The hooks are listed in `.pre-commit-hooks.yaml`, and their Python entry points
are also listed in `pyproject.toml`. Their implementations are in
`src/rapids_pre_commit_hooks`.

Follow the existing code conventions that you see in the existing hooks. These
include, but are not limited to:

- Use the `LintMain` class to drive the hook.
- Add warnings, notes, and replacements to a `Linter` rather than reading and
  writing files directly.
- Warnings and notes should be all lower-case with no period at the end.
- Type annotations should be applied to function arguments, return types, and
  class fields.

Hooks should be idempotent: applying their fixes should cause the associated
warning to not be raised on subsequent runs. Fixes should only be applied when
the fix is clear and unambiguous. If there is ambiguity about what should be
done, it should be left to the developer to decide how to fix the problem.

Docstrings are generally not used except for large and/or complicated functions
whose purpose may not be obvious. See
`rapids_pre_commit_hooks_test_utiles.parse_named_spans()` as an example of such
an exception.

## `dependencies.yaml` hooks
`dependencies.yaml` is the file format used by
[rapids-dependency-file-generator](https://github.com/rapidsai/dependency-file-generator).
DFG has a JSON schema specifying the format of the YAML file.

Almost all functionality related to checking `dependencies.yaml` should go in
the existing `verify-dependencies` check. `verify-alpha-spec` exists separately
for historical reasons and because its purpose is different from
`verify-dependencies`. In any case, anything that traverses `dependencies.yaml`
should use
`rapids_pre_commit_hooks.utils.dependencies_yaml.traverse_dependencies_yaml()`.
New sub-checks that get added to `verify-dependencies` should have new modules
created in `rapids_pre_commit_hooks.dependencies`, and the check functions
should be added to `rapids_pre_commit_hooks.dependencies.check_dependencies()`.

# Tests
All tests can be found in the `tests` directory.

## Unit Tests and Integration Tests
Unit tests and some integration tests can be found in
`tests/rapids_pre_commit_hooks`. These tests make heavy use of
`rapids_pre_commit_hooks_test_utils.parse_named_spans()`. This function
provides a DSL that is meant for easily picking spans (`tuple[int, int]`) out
of a string of text rather than hard-coding the integers. All tests that use
spans should use `parse_named_spans()`, except tests that test the testing
infrastructure itself. See the documentation for `parse_named_spans()` for
a description of the DSL, and see the existing tests for examples.

Parameterized tests should generally only have one dimension. Parameters
should be listed as an array rather than as a single string. Each parameterized
entry should be declared with `pytest.param()` and a sensible `id` named in
kebab-case (`name-of-test`).

In most cases, `zip_expected_warnings()` should be used to construct a list of
expected warnings from the named spans and the parallel warning metadata, both
of which should come from the test parameterization.

Test files generally do not require type annotations.

When in doubt, follow the existing code conventions present in the tests.

### YAML Tests
Tests that operate on YAML text should use
`rapids_pre_commit_hooks_test_utils.find_yaml_node_for_span()` rather than
hard-coding node paths.

## Smoke Tests
`tests/examples` contains smoke tests for each hook. Each hook directory
contains a `pass/main` directory and a `fail/main` directory, respectively
containing repository contents that will pass and fail the respective hook.
The `verify-copyright` hook also contains `pass/branch` and `fail/branch` due
to its special Git requirements, but most hooks don't need this. Some smoke
tests also have a `metadata.yaml` file with additional information. See
`tests/test_pre_commit.py` for the smoke test implementation.

If a hook does not have a corresponding smoke test, `tests/test_pre_commit.py`
will fail.
