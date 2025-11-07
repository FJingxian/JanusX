#!/bin/bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
MODULE_DIR="$SCRIPT_DIR/module"
VENV_PYLAUCH="$SCRIPT_DIR/.venv/bin/python"

MODULE_NAME=$1
MODULE=${MODULE_NAME/.py/}

if [ $# -eq 0 ] || [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    echo "Usage: $0 <module> [parameter]"
    echo "Modules:" $(ls $MODULE_DIR | grep -v "_")
    exit 0
fi

if [ -f "$MODULE_DIR/$MODULE.py" ];then
    shift
    $VENV_PYLAUCH -u $MODULE_DIR/$MODULE.py $@
else
    echo "Unkwown module: $MODULE;" "Installed modules:" $(ls $MODULE_DIR | grep -v "_")
fi