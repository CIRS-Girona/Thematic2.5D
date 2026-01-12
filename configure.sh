#!/bin/bash

sudo apt update
sudo apt install -y build-essential git python3-venv libeigen3-dev

# Create and activate virtual environment
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Create deps if it doesn't exist
mkdir -p deps
cd deps

# Install fast-slic
if [ ! -d "fast-slic" ]; then
    git clone https://github.com/m-krastev/fast-slic.git
fi
cd fast-slic
pip install .

# Compile C++ bindings
cd ../../src/cpp/
g++ -O3 -fopenmp -shared -fPIC -o libfastmetrics.so image_metrics.cpp
g++ -O3 -fopenmp -shared -fPIC -o libfastfeatures.so feature_extraction.cpp
mv *.so ../../deps/

cd ../../