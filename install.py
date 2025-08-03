import subprocess
import sys
import os
from pathlib import Path
import platform
import time

def print_step(step: str):
    """Print a formatted step message"""
    print(f"\n{'='*80}\n{step}\n{'='*80}")

def run_command(command: list, description: str, ignore_errors: bool = False):
    """Run a command with error handling"""
    print_step(description)
    try:
        subprocess.run(command, check=not ignore_errors)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        if not ignore_errors:
            raise
        return False

def get_python_executable():
    """Get the correct Python executable based on platform"""
    if platform.system() == "Windows":
        return ".\\venv\\Scripts\\python"
    return "./venv/bin/python"

def get_pip_executable():
    """Get the correct pip executable based on platform"""
    if platform.system() == "Windows":
        return ".\\venv\\Scripts\\pip"
    return "./venv/bin/pip"

def setup_virtual_environment():
    """Create and activate virtual environment"""
    print_step("Setting up virtual environment")
    
    # Remove existing venv if it exists
    if os.path.exists("venv"):
        if platform.system() == "Windows":
            subprocess.run("rmdir /s /q venv", shell=True)
        else:
            subprocess.run("rm -rf venv", shell=True)
    
    # Create new venv
    subprocess.run([sys.executable, "-m", "venv", "venv"], check=True)
    
    # Upgrade pip
    pip = get_pip_executable()
    subprocess.run([pip, "install", "--upgrade", "pip"], check=True)

def install_core_dependencies():
    """Install core dependencies first"""
    pip = get_pip_executable()
    
    # Core dependencies
    print_step("Installing core dependencies")
    core_deps = [
        "fastapi==0.68.1",
        "uvicorn==0.15.0",
        "python-dotenv==0.19.0",
        "sqlalchemy==1.4.23",
        "pymysql==1.0.2",
        "pillow==9.5.0",
        "numpy==1.24.3",
        "requests==2.31.0"
    ]
    run_command([pip, "install"] + core_deps, "Installing core packages")

def install_build_dependencies():
    """Install build dependencies"""
    pip = get_pip_executable()
    
    # Build dependencies
    print_step("Installing build dependencies")
    build_deps = [
        "setuptools>=65.5.1",
        "wheel>=0.38.4",
        "cmake>=3.26.3",
        "ninja>=1.11.1"
    ]
    run_command([pip, "install"] + build_deps, "Installing build dependencies")

def install_ml_dependencies():
    """Install ML/DL dependencies"""
    pip = get_pip_executable()
    
    # PyTorch and related packages
    print_step("Installing PyTorch and related packages")
    torch_command = [pip, "install"]
    if platform.system() == "Windows":
        torch_command.extend([
            "torch==2.0.1+cpu",
            "torchvision==0.15.2+cpu",
            "--index-url", "https://download.pytorch.org/whl/cpu"
        ])
    else:
        torch_command.extend([
            "torch==2.0.1",
            "torchvision==0.15.2"
        ])
    run_command(torch_command, "Installing PyTorch")
    
    # Other ML packages
    ml_deps = [
        "transformers==4.30.2",
        "sentence-transformers==2.2.2",
        "scikit-learn==1.3.0",
        "opencv-python-headless==4.8.0.74",
        "timm==0.9.2",
        "efficientnet-pytorch==0.7.1"
    ]
    run_command([pip, "install"] + ml_deps, "Installing ML packages")

def install_nlp_dependencies():
    """Install NLP dependencies"""
    pip = get_pip_executable()
    python = get_python_executable()
    
    # SpaCy and related packages
    print_step("Installing NLP packages")
    nlp_deps = [
        "spacy==3.6.0",
        "spacy-transformers==1.2.5",
        "thefuzz==0.19.0",
        "python-Levenshtein==0.21.1"
    ]
    run_command([pip, "install"] + nlp_deps, "Installing NLP packages")
    
    # Download spaCy model
    print_step("Downloading spaCy model")
    run_command(
        [python, "-m", "spacy", "download", "en_core_web_lg"],
        "Downloading spaCy model"
    )

def install_remaining_dependencies():
    """Install remaining dependencies"""
    pip = get_pip_executable()
    
    # Database and search
    print_step("Installing database and search packages")
    db_deps = [
        "faiss-cpu==1.7.4",
        "pandas==2.0.3"
    ]
    run_command([pip, "install"] + db_deps, "Installing database packages")
    
    # Monitoring and visualization
    print_step("Installing monitoring packages")
    monitoring_deps = [
        "wandb==0.15.4",
        "matplotlib==3.7.1",
        "seaborn==0.12.2",
        "prometheus-client==0.17.1",
        "tqdm==4.65.0"
    ]
    run_command([pip, "install"] + monitoring_deps, "Installing monitoring packages")
    
    # Optional dependencies
    print_step("Installing optional packages")
    optional_deps = [
        "scikit-image==0.21.0",
        "albumentations==1.3.1"
    ]
    run_command([pip, "install"] + optional_deps, "Installing optional packages", ignore_errors=True)
    
    # Pre-trained models
    print_step("Installing pre-trained models")
    model_deps = [
        "git+https://github.com/openai/CLIP.git"
    ]
    run_command([pip, "install"] + model_deps, "Installing pre-trained models")

def verify_installation():
    """Verify the installation"""
    python = get_python_executable()
    
    print_step("Verifying installation")
    verification_code = """
import torch
import transformers
import spacy
import faiss
import cv2
print("All critical packages imported successfully!")
"""
    
    with open("verify_install.py", "w") as f:
        f.write(verification_code)
    
    try:
        subprocess.run([python, "verify_install.py"], check=True)
        print("\nInstallation verified successfully!")
    except subprocess.CalledProcessError:
        print("\nWarning: Some packages may not have installed correctly.")
    finally:
        if os.path.exists("verify_install.py"):
            os.remove("verify_install.py")

def main():
    try:
        start_time = time.time()
        
        # Setup steps
        setup_virtual_environment()
        install_build_dependencies()
        install_core_dependencies()
        install_ml_dependencies()
        install_nlp_dependencies()
        install_remaining_dependencies()
        verify_installation()
        
        duration = time.time() - start_time
        print(f"\nInstallation completed in {duration:.1f} seconds")
        
        print("\nTo activate the virtual environment:")
        if platform.system() == "Windows":
            print("    .\\venv\\Scripts\\activate")
        else:
            print("    source venv/bin/activate")
        
    except Exception as e:
        print(f"\nError during installation: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 