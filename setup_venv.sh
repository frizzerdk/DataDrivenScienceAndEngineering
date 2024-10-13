#!/bin/bash

# Install system dependencies
sudo apt-get update
sudo apt-get install -y python3-tk python3-pil python3-pil.imagetk

# Remove existing virtual environment if it exists
if [ -d "venv" ]; then
    echo "Removing existing virtual environment..."
    rm -rf venv
fi

# Create a new virtual environment with system site packages
python3 -m venv venv --system-site-packages

# Activate the virtual environment
source venv/bin/activate

# Upgrade pip within the virtual environment
pip install --upgrade pip

# Install dependencies from requirements.txt
pip install -r requirements.txt

# Verify installations
python -c "import matplotlib; print(matplotlib.get_backend())"
python -c "from PIL import ImageTk; print('ImageTk successfully imported')"

echo "Setup complete. Virtual environment 'venv' created with system site packages and dependencies installed."
echo "Don't forget to activate the virtual environment with 'source venv/bin/activate' before running your scripts."
