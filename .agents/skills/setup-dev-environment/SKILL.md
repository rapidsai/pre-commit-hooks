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
pip install -e .[test]
```

Both the hooks and the tests also require git to work correctly. Git is not
available as a Python wheel, and has to be installed separately from
conda-forge (using `conda` or `mamba`) or from the package manager.

Once the environment is set up, `pytest` can be used to run the tests.
