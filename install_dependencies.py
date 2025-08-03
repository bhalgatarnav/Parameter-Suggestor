import subprocess
import sys
import os
from typing import List

def run_pip_install(packages: List[str], upgrade_pip: bool = False):
    """Run pip install with the specified packages"""
    try:
        if upgrade_pip:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + packages)
        print(f"Successfully installed {', '.join(packages)}")
    except subprocess.CalledProcessError as e:
        print(f"Error installing packages: {e}")
        sys.exit(1)

def main():
    # Ensure we're in a virtual environment
    if not hasattr(sys, 'real_prefix') and not (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("Please activate your virtual environment first!")
        sys.exit(1)
    
    print("Starting staged dependency installation...")
    
    # Stage 1: Upgrade pip and install build tools
    print("\nStage 1: Installing build tools...")
    run_pip_install(["pip", "setuptools", "wheel"], upgrade_pip=True)
    
    # Stage 2: Install torch and related packages
    print("\nStage 2: Installing PyTorch...")
    run_pip_install([
        "torch==2.0.1",
        "torchvision==0.15.2"
    ])
    
    # Stage 3: Install huggingface packages
    print("\nStage 3: Installing Hugging Face packages...")
    run_pip_install([
        "huggingface-hub==0.16.4",
        "transformers==4.31.0",
        "diffusers==0.21.4",
        "accelerate==0.21.0",
        "safetensors==0.3.1"
    ])
    
    # Stage 4: Install ControlNet and related packages
    print("\nStage 4: Installing ControlNet dependencies...")
    run_pip_install([
        "controlnet-aux==0.0.6",
        "xformers==0.0.20",
        "opencv-python==4.8.0.74"
    ])
    
    # Stage 5: Install remaining dependencies
    print("\nStage 5: Installing remaining dependencies...")
    run_pip_install([
        "numpy>=1.19.2",
        "Pillow>=8.0.0",
        "tqdm>=4.49.0",
        "streamlit>=1.24.0",
        "fastapi>=0.68.0",
        "uvicorn>=0.15.0",
        "python-dotenv>=0.19.0",
        "sqlalchemy>=1.4.23",
        "requests>=2.26.0",
        "openai>=0.27.0",
        "scikit-learn>=0.24.2",
        "pandas>=1.3.0",
        "pydantic>=1.8.2",
        "python-multipart>=0.0.5",
        "aiofiles>=0.7.0",
        "python-jose>=3.3.0",
        "passlib>=1.7.4",
        "bcrypt>=3.2.0",
        "email-validator>=1.1.3"
    ])
    
    print("\nDependency installation completed successfully!")

if __name__ == "__main__":
    main() 