#!/bin/bash
# Runs pre-commit

set -ue

pip install pre-commit

python -m pre-commit run -a
