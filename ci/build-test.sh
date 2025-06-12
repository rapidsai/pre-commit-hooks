#!/bin/bash
# Builds and tests Python package

set -ue

# According to https://pre-commit.com/#python,
# pre-commit will install the package with 'pip install .' from the repo root.
#
# So this installs it the same way.
pip install \
  --extra-index-url=https://pypi.anaconda.org/rapidsai-wheels-nightly/simple \
  .[test]

pytest
