# pre-commit-hooks

This repository contains [pre-commit](https://pre-commit.com) hooks used by RAPIDS projects.

## Using hooks

Copy the following into your repository's `.pre-commit-config.yaml`:

```yaml
- repo: https://github.com/rapidsai/pre-commit-hooks
  rev: v0.0.1  # Use the ref you want to point at
  hooks:  # Hook names to use
    - id: verify-copyright
```

## Suppressing false positives

Some checks are prone to false positives, particularly `verify-hardcoded-version`. In such
cases, you can use `rapids-pre-commit-hooks: disable`, `rapids-pre-commit-hooks: enable`, and
`rapids-pre-commit-hooks: disable-next-line` directives:

```python
def deprecated_function():
    # rapids-pre-commit-hooks: disable[verify-hardcoded-version]
    """Do some deprecated stuff

    .. deprecated:: Deprecated in 26.04
    """
    # rapids-pre-commit-hooks: enable[verify-hardcoded-version]

    # rapids-pre-commit-hooks: disable-next-line[verify-hardcoded-version]
    warnings.warn("deprecated_function() has been deprecated since 26.04")
```

## Included hooks

All hooks are listed in `.pre-commit-hooks.yaml`.

## Acknowledgements

This project uses Bashlex.

- PyPI: https://pypi.org/project/bashlex/
- GitHub: https://github.com/idank/bashlex
