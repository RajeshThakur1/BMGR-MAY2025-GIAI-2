#!/usr/bin/env python3
"""
Install requirements for Gradio tutorial
"""

import subprocess
import sys

def install_package(package):
    """Install a package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✅ {package} installed successfully")
        return True
    except subprocess.CalledProcessError:
        print(f"❌ Failed to install {package}")
        return False

def main():
    print("🔧 Installing Gradio Tutorial Requirements")
    print("=" * 50)
    
    packages = [
        "gradio",
        "pandas", 
        "numpy",
        "matplotlib",
        "pillow"
    ]
    
    print(f"Installing {len(packages)} packages...")
    
    success_count = 0
    for package in packages:
        print(f"\nInstalling {package}...")
        if install_package(package):
            success_count += 1
    
    print(f"\n📊 Installation Summary:")
    print(f"✅ Successfully installed: {success_count}/{len(packages)} packages")
    
    if success_count == len(packages):
        print("\n🎉 All packages installed! You can now run the tutorial:")
        print("python gradio_step_by_step.py")
        print("or")
        print("python run_gradio_examples.py")
    else:
        print(f"\n⚠️  Some packages failed to install. Please install them manually:")
        print("pip install gradio pandas numpy matplotlib pillow")

if __name__ == "__main__":
    main() 