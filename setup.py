import os
import sys
import subprocess
import platform

def setup_environment():
    """Set up the development environment"""
    print("Setting up Material Advisor development environment...")
    
    # Determine the Python executable
    python_cmd = sys.executable
    pip_cmd = f"{python_cmd} -m pip"
    
    # Create virtual environment if it doesn't exist
    venv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "backend_env")
    if not os.path.exists(venv_path):
        print("\nCreating virtual environment...")
        subprocess.run(f"{python_cmd} -m venv {venv_path}", shell=True, check=True)
    
    # Activate virtual environment
    if platform.system() == "Windows":
        activate_script = os.path.join(venv_path, "Scripts", "activate.bat")
        pip_path = os.path.join(venv_path, "Scripts", "pip.exe")
    else:
        activate_script = os.path.join(venv_path, "bin", "activate")
        pip_path = os.path.join(venv_path, "bin", "pip")
    
    # Upgrade pip
    print("\nUpgrading pip...")
    subprocess.run(f"{pip_path} install --upgrade pip", shell=True, check=True)
    
    # Install dependencies
    print("\nInstalling dependencies...")
    requirements_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "requirements.txt")
    subprocess.run(f"{pip_path} install -r {requirements_path}", shell=True, check=True)

    print("\nEnvironment setup complete!")
    print("\nTo activate the environment:")
    if platform.system() == "Windows":
        print(f"Run: {venv_path}\\Scripts\\activate")
    else:
        print(f"Run: source {venv_path}/bin/activate")

if __name__ == "__main__":
    try:
        setup_environment()
    except subprocess.CalledProcessError as e:
        print(f"\nError during setup: {str(e)}")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {str(e)}")
        sys.exit(1)