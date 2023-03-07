# pre-commit-hooks

This repository contains [pre-commit](https://pre-commit.com) hooks used by RAPIDS projects.

## Using hooks

Copy the following into your repository's `.pre-commit-config.yaml`:

```yaml
- repo: https://github.com/rapidsai/pre-commit-hooks
  rev: v0.0.1  # Use the ref you want to point at
  hooks:
    - id: copyright-checker  # Hook names
```
## Included hooks

All hooks are listed in `.pre-commit-hooks.yaml`.

- (No hooks exist yet)
