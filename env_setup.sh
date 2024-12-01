#!/bin/bash

# Define variables
source "./config.sh"

if ! command -v conda &> /dev/null; then
    echo "Conda command not found. Please install Anaconda or Miniconda."
    exit 1
fi
source "$(conda info --base)/etc/profile.d/conda.sh"

create_conda_env() {
    echo "Creating conda environment '$ENV_NAME' with Python $PYTHON_VERSION..."
    if conda create -n "$ENV_NAME" python="$PYTHON_VERSION" -y; then
        echo "Conda environment '$ENV_NAME' created successfully."
    else
        echo "Failed to create conda environment '$ENV_NAME'."
        exit 1
    fi
}

install_pytorch() {
    echo "Installing PyTorch $PYTORCH_VERSION in the conda environment..."
    if conda activate "$ENV_NAME"; then
        if conda install pytorch=="$PYTORCH_VERSION" torchvision torchaudio -c pytorch -y; then
            echo "PyTorch installed successfully."
        else
            echo "Failed to install PyTorch."
            exit 1
        fi
    else
        echo "Failed to activate conda environment '$ENV_NAME'."
        exit 1
    fi
    conda deactivate
}

install_dependencies() {
    echo "Installing additional dependencies..."
    if conda activate "$ENV_NAME"; then
        if conda install tqdm pandas lxml -y && \
           conda install conda-forge::transformers -y; then
            echo "Additional dependencies installed successfully."
        else
            echo "Failed to install additional dependencies."
            exit 1
        fi
    else
        echo "Failed to activate conda environment '$ENV_NAME'."
        exit 1
    fi
    conda deactivate
}

setup_environment() {
    echo "Setting up environment..."

    create_conda_env
    install_pytorch
    install_dependencies

    echo "Environment setup complete! Activate it using 'conda activate $ENV_NAME'."
    echo "Deactivate using 'conda deactivate'."
}

# Execute the setup
setup_environment
