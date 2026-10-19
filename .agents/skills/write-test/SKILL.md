---
name: write-test
description: Write tests for rapids-pre-commit-hooks.
---
# Test Guidelines
All tests can be found in the `tests` directory.

## Unit Tests and Integration Tests
The tests for rapids-pre-commit-hooks make heavy use of
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
