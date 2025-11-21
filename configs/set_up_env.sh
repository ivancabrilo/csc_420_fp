#!/bin/bash

# Create a Python virtual environment
echo "Creating Python venv"
python3 -m venv csc_420_env

# Activate the virtual environment
source csc_420_env/bin/activate 

# Upgrade pip to latest version!!!
echo "Upgrading pip..."
pip install --upgrade pip

# Install all dependencies from requirements.txt
echo "Installing dependencies"
pip install -r requirements.txt

echo "Environment setup complete!"

# Deactivate the virtual environment
deactivate