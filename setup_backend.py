#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Setup Script สำหรับ Thammasat AI Agent Backend
ใช้สำหรับติดตั้งและตั้งค่า backend อัตโนมัติ
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def print_banner():
    """แสดง banner สำหรับ setup script"""
    print("=" * 60)
    print("🚀 Thammasat AI Agent Backend Setup")
    print("=" * 60)
    print()

def check_python_version():
    """ตรวจสอบ Python version"""
    print("🐍 ตรวจสอบ Python version...")
    
    if sys.version_info < (3, 8):
        print("❌ Error: Python 3.8+ is required")
        print(f"   Current version: {sys.version}")
        return False
    
    print(f"✅ Python {sys.version.split()[0]} - OK")
    return True

def create_virtual_environment():
    """สร้าง virtual environment"""
    print("\n📦 สร้าง Virtual Environment...")
    
    venv_path = Path("venv")
    
    if venv_path.exists():
        print("⚠️  Virtual environment มีอยู่แล้ว - ข้าม")
        return True
    
    try:
        subprocess.run([sys.executable, "-m", "venv", "venv"], check=True)
        print("✅ Virtual environment สร้างเสร็จแล้ว")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error creating virtual environment: {e}")
        return False

def get_pip_command():
    """หา pip command ที่เหมาะสม"""
    if os.name == "nt":  # Windows
        return "venv\\Scripts\\pip"
    else:  # Unix/Linux/macOS
        return "venv/bin/pip"

def install_dependencies():
    """ติดตั้ง dependencies จาก requirements.txt"""
    print("\n📚 ติดตั้ง Dependencies...")
    
    pip_cmd = get_pip_command()
    
    if not Path("requirements.txt").exists():
        print("❌ Error: requirements.txt not found")
        return False
    
    try:
        # Upgrade pip first
        print("   Upgrading pip...")
        subprocess.run([pip_cmd, "install", "--upgrade", "pip"], check=True)
        
        # Install requirements
        print("   Installing requirements...")
        subprocess.run([pip_cmd, "install", "-r", "requirements.txt"], check=True)
        
        print("✅ Dependencies ติดตั้งเสร็จแล้ว")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing dependencies: {e}")
        return False

def setup_environment_file():
    """ตั้งค่า environment file"""
    print("\n🔧 ตั้งค่า Environment File...")
    
    env_example = Path(".env.example")
    env_file = Path(".env")
    
    if env_file.exists():
        print("⚠️  .env file มีอยู่แล้ว - ข้าม")
        return True
    
    if env_example.exists():
        try:
            shutil.copy(env_example, env_file)
            print("✅ สร้าง .env file จาก .env.example")
            print("📝 กรุณาแก้ไข .env file และใส่ API keys ของคุณ")
            return True
        except Exception as e:
            print(f"❌ Error copying .env.example: {e}")
            return False
    else:
        # สร้าง .env file พื้นฐาน
        try:
            with open(env_file, "w", encoding="utf-8") as f:
                f.write("# Thammasat AI Agent Environment Variables\n")
                f.write("GOOGLE_API_KEY=your_google_api_key_here\n")
                f.write("TAVILY_API_KEY=your_tavily_api_key_here\n")
                f.write("API_HOST=0.0.0.0\n")
                f.write("API_PORT=8000\n")
                f.write("ENVIRONMENT=development\n")
                f.write("LOG_LEVEL=INFO\n")
            
            print("✅ สร้าง .env file พื้นฐาน")
            print("📝 กรุณาแก้ไข .env file และใส่ API keys ของคุณ")
            return True
        except Exception as e:
            print(f"❌ Error creating .env file: {e}")
            return False

def create_directories():
    """สร้าง directories ที่จำเป็น"""
    print("\n📁 สร้าง Directories...")
    
    directories = [
        "logs",
        "uploads",
        "data",
        "tests"
    ]
    
    for directory in directories:
        dir_path = Path(directory)
        if not dir_path.exists():
            try:
                dir_path.mkdir(parents=True)
                print(f"✅ สร้าง directory: {directory}")
            except Exception as e:
                print(f"❌ Error creating directory {directory}: {e}")
                return False
        else:
            print(f"⚠️  Directory {directory} มีอยู่แล้ว - ข้าม")
    
    return True

def create_gitignore():
    """สร้าง .gitignore file"""
    print("\n📝 สร้าง .gitignore...")
    
    gitignore_path = Path(".gitignore")
    
    if gitignore_path.exists():
        print("⚠️  .gitignore มีอยู่แล้ว - ข้าม")
        return True
    
    gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environment
venv/
env/
ENV/

# Environment Variables
.env
.env.local
.env.production

# IDE
.vscode/
.idea/
*.swp
*.swo

# Logs
logs/
*.log

# Uploads
uploads/
*.pdf
*.docx

# Data
data/
*.db
*.sqlite

# OS
.DS_Store
Thumbs.db

# Testing
.pytest_cache/
.coverage
htmlcov/

# FastAPI
.pytest_cache/
"""
    
    try:
        with open(gitignore_path, "w", encoding="utf-8") as f:
            f.write(gitignore_content)
        print("✅ .gitignore สร้างเสร็จแล้ว")
        return True
    except Exception as e:
        print(f"❌ Error creating .gitignore: {e}")
        return False

def verify_setup():
    """ตรวจสอบการติดตั้ง"""
    print("\n🔍 ตรวจสอบการติดตั้ง...")
    
    # ตรวจสอบ virtual environment
    venv_path = Path("venv")
    if not venv_path.exists():
        print("❌ Virtual environment not found")
        return False
    
    # ตรวจสอบ .env file
    env_path = Path(".env")
    if not env_path.exists():
        print("❌ .env file not found")
        return False
    
    # ตรวจสอบ main.py
    main_path = Path("main.py")
    if not main_path.exists():
        print("❌ main.py not found")
        return False
    
    print("✅ การติดตั้งเสร็จสมบูรณ์")
    return True

def show_next_steps():
    """แสดงขั้นตอนต่อไป"""
    print("\n🎯 ขั้นตอนต่อไป:")
    print("=" * 40)
    print("1. แก้ไข .env file และใส่ API keys ของคุณ:")
    print("   - GOOGLE_API_KEY (จำเป็น)")
    print("   - TAVILY_API_KEY (ไม่บังคับ)")
    print()
    print("2. เปิดใช้งาน virtual environment:")
    if os.name == "nt":  # Windows
        print("   venv\\Scripts\\activate")
    else:  # Unix/Linux/macOS
        print("   source venv/bin/activate")
    print()
    print("3. รัน server:")
    print("   python main.py")
    print()
    print("4. ทดสอบ API:")
    print("   เปิด http://localhost:8000/docs")
    print()
    print("🚀 พร้อมใช้งาน Thammasat AI Agent Backend!")

def main():
    """Main setup function"""
    print_banner()
    
    success = True
    
    # ตรวจสอบ Python version
    if not check_python_version():
        success = False
    
    # สร้าง virtual environment
    if success and not create_virtual_environment():
        success = False
    
    # ติดตั้ง dependencies
    if success and not install_dependencies():
        success = False
    
    # ตั้งค่า environment file
    if success and not setup_environment_file():
        success = False
    
    # สร้าง directories
    if success and not create_directories():
        success = False
    
    # สร้าง .gitignore
    if success and not create_gitignore():
        success = False
    
    # ตรวจสอบการติดตั้ง
    if success and not verify_setup():
        success = False
    
    print("\n" + "=" * 60)
    
    if success:
        print("🎉 Setup เสร็จสมบูรณ์!")
        show_next_steps()
    else:
        print("❌ Setup ไม่สำเร็จ - กรุณาตรวจสอบ errors ข้างต้น")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)