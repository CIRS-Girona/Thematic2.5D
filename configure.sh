#!/bin/bash

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Create deps if it doesn't exist
mkdir -p deps
cd deps

# Check if repo wasn't cloned already. If not, clone it
if [ ! -d "fast-slic" ]; then
    git clone https://github.com/m-krastev/fast-slic.git
fi
cd fast-slic

pip install .