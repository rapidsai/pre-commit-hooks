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
## Included hooks

All hooks are listed in `.pre-commit-hooks.yaml`.

## Acknowledgements

This project uses Bashlex.

- PyPI: https://pypi.org/project/bashlex/
- GitHub: https://github.com/idank/bashlex
