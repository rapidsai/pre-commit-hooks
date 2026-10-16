---
name: setup-dev-environment
description: >
  Create a development environment suitable for developing and testing
  rapids-pre-commit-hooks.
---
# Overview
rapids-pre-commit-hooks is a standard Python project. All of the Python
dependencies needed for testing can be installed by running:

```sh
python -m venv .venv
. .venv/bin/activate
pip install -e '.[test]'
```

Once the environment is set up, `pytest` can be used to run the tests.
