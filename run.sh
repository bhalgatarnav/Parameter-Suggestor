#!/bin/bash

if [ -z "$1" ]; then
    echo "Usage: ./run.sh [backend|frontend]"
    exit 1
fi

if [ ! -d "venv" ]; then
    echo "Virtual environment not found. Running installation..."
    python3 scripts/install.py
    if [ $? -ne 0 ]; then
        echo "Installation failed."
        exit 1
    fi
fi

source venv/bin/activate
python scripts/run.py "$1" 