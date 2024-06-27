#!/bin/bash
# Builds and tests Python package

set -ue

pip install build

python -m build .

for PKG in dist/*; do
  echo "$PKG"
  pip uninstall -y rapids-pre-commit-hooks
  pip install --extra-index-url=https://pypi.anaconda.org/rapidsai-wheels-nightly/simple "$PKG[test]"
  pytest
done
