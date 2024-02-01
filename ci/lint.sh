#!/bin/bash
# Runs pre-commit

set -ue

pip install pre-commit

pre-commit run --all-files --show-diff-on-failure
