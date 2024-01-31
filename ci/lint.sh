#!/bin/bash
# Runs pre-commit

set -ue

pip install pre-commit

pre-commit run -a
