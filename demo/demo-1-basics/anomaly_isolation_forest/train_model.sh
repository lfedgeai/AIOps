#!/bin/bash

echo "Installing dependencies..."
pip3 install -r requirements.txt

echo "Setting PYTHONPATH..."
export PYTHONPATH=$(pwd)


echo "Running anomaly model Python script..."
python3 -m app.scripts.train_model
# ./app/scripts/train_model.py
