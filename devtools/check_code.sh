#!/bin/bash

set -e

# Disabled pylint messages
# C0114, C0116: Missing module docstring (missing-module-docstring)
# C0209: Formatting a regular string which could be a f-string (consider-using-f-string)
# C0103: Variable name doesn't conform to snake_case naming style (invalid-name)
# R0801: Similar lines in %s files %s
# W0621: Redefining name from outer scope
# W1514: Using open without explicitly specifying an encoding (unspecified-encoding)
# R0912: Too many branches (too-many-branches)
# R0913: Too many arguments
# R0914: Too many local variables (too-many-locals)
# R0915: Too many statements (too-many-statements)
# R1702: Too many nested blocks (too-many-nested-blocks)
# R0902: Too many instance attributes
# R0903: Too few public methods
# W0221: Arguments differ
# C0415: Import outside toplevel
# C0302: Too many lines in module
# W0718: Catching too general exception Exception
# R0902: Too many instance attributes
# R1702: Too many nested blocks
PYLINT_DISABLED="C0114,C0116,C0209,C0103,R0801,W0621,W1514,R0912,R0913,R0914,R0915,R1702"
PYLINT_DISABLED+=",R0902,R0903,W0221,C0415,C0302,R0401,W0718,R0902,R1702"

readonly MAX_LINE_LENGTH=100
readonly INDENT_SPACES=4

##########################################################################################
# Functions
#

Usage() {
    echo "Check codebase with "
    echo "Usage:"
    echo "$0 [-c] [-h]"
    echo
    echo "-c: Clear cache before invoking PyTest"
    echo "-h: Print this help message"
    echo
    echo "$0"
}


##########################################################################################
# Main
#
clear_cache=0
while getopts ":hc" option; do
  case "${option}" in
    c ) clear_cache=1;;
    h ) Usage; exit;;
    \? ) Usage; exit;;
    : ) # Missing required argument
        Usage; exit;;
  esac
done

# PEP8
echo "Flake8 check:"
flake8 \
  --max-line-length=${MAX_LINE_LENGTH} \
  --indent-size=${INDENT_SPACES} \
  --ignore=E121,E123,E126,E226,E24,E704,W503,W504,W605,E203 \
  --extend-exclude '_*' \
  docling_ibm_models/ tests/
echo "Flake8 - OK"
echo

# # Pylint
# echo "Pylint check:"
# indent_string=$(printf '%*s' ${INDENT_SPACES} "" | tr ' ' 'n' | tr 'n' ' ')
# # echo "indent_string: '${indent_string}'"
# pylint \
#   --max-line-length ${MAX_LINE_LENGTH} \
#   --indent-string "${indent_string}" \
#   --disable ${PYLINT_DISABLED} \
#   --extension-pkg-whitelist='pydantic' \
#   --ignore-patterns '[!_]' \
#   docling_ibm_models/ tests/
#
# echo "Pylint check - OK"
# echo

# Unit tests with PyTest
echo "PyTest:"
if [ ${clear_cache} -eq 1 ]; then
  echo "Clear pytest cache first"
  echo
  python -m pytest -n auto --cache-clear --ignore=docling_ibm_models/ tests/
else
  python -m pytest -n auto --ignore=docling_ibm_models/ tests/
fi
echo "PyTest check - OK"
