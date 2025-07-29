#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Startup Script สำหรับ Thammasat AI Agent Backend
รันผ่าน script นี้เพื่อให้มีการตรวจสอบและ configuration อัตโนมัติ
"""

import os
import sys
import subprocess
from pathlib import Path
from dotenv import load_dotenv

def print_banner():
    """แสดง banner เมื่อเริ่มต้น server"""
    print("=" * 60)
    print("🚀 Starting Thammasat AI Agent Backend")
    print("=" * 60)

def check_environment():
    """ตรวจสอบ environment และ configuration"""
    print("🔍 ตรวจสอบ Environment...")
    
    # ตรวจสอบ .env file
    env_path = Path(".env")
    if not env_path.exists():
        print("❌ Error: .env file not found")
        print("   กรุณารัน: python setup_backend.py")
        return False
    
    # โหลด environment variables
    load_dotenv()
    
    # ตรวจสอบ API keys ที่จำเป็น
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key or google_api_key == "your_google_api_key_here":
        print("❌ Error: GOOGLE_API_KEY ไม่ได้ตั้งค่าใน .env file")
        print("   กรุณาใส่ Google AI API key ใน .env file")
        return False
    
    print("✅ GOOGLE_API_KEY - OK")
    
    # ตรวจสอบ Tavily API key (ไม่บังคับ)
    tavily_api_key = os.getenv("TAVILY_API_KEY")
    if not tavily_api_key or tavily_api_key == "your_tavily_api_key_here":
        print("⚠️  TAVILY_API_KEY ไม่ได้ตั้งค่า - การค้นหาเว็บจะไม่พร้อมใช้งาน")
    else:
        print("✅ TAVILY_API_KEY - OK")
    
    return True

def check_dependencies():
    """ตรวจสอบ dependencies ที่จำเป็น"""
    print("\n📦 ตรวจสอบ Dependencies...")
    
    required_packages = [
        "fastapi",
        "uvicorn",
        "google.generativeai",
        "pandas",
        "pydantic"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
        print("   กรุณาติดตั้ง: pip install -r requirements.txt")
        return False
    
    return True

def create_required_directories():
    """สร้าง directories ที่จำเป็น"""
    print("\n📁 ตรวจสอบ Directories...")
    
    directories = ["logs", "uploads", "data"]
    
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
            print(f"✅ {directory} directory exists")
    
    return True

def get_server_config():
    """ดึง configuration สำหรับ server"""
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    reload = os.getenv("ENVIRONMENT", "development") == "development"
    log_level = os.getenv("LOG_LEVEL", "info").lower()
    
    return {
        "host": host,
        "port": port,
        "reload": reload,
        "log_level": log_level
    }

def start_server():
    """เริ่มต้น FastAPI server"""
    print("\n🚀 Starting FastAPI Server...")
    
    config = get_server_config()
    
    print(f"   Host: {config['host']}")
    print(f"   Port: {config['port']}")
    print(f"   Reload: {config['reload']}")
    print(f"   Log Level: {config['log_level']}")
    print(f"\n📖 API Documentation: http://localhost:{config['port']}/docs")
    print(f"🔄 Health Check: http://localhost:{config['port']}/health")
    print("\n" + "=" * 60)
    print("🎉 Server is starting... (กด Ctrl+C เพื่อหยุด)")
    print("=" * 60)
    
    try:
        import uvicorn
        from main import app
        
        uvicorn.run(
            app,
            host=config["host"],
            port=config["port"],
            reload=config["reload"],
            log_level=config["log_level"]
        )
    except KeyboardInterrupt:
        print("\n\n👋 Server stopped by user")
    except Exception as e:
        print(f"\n❌ Error starting server: {e}")
        return False
    
    return True

def show_usage():
    """แสดงวิธีการใช้งาน"""
    print("Usage: python run_server.py [OPTIONS]")
    print("\nOptions:")
    print("  --help, -h     แสดงคำแนะนำการใช้งาน")
    print("  --check, -c    ตรวจสอบ environment เท่านั้น (ไม่รัน server)")
    print("  --setup, -s    รัน setup script")
    print("\nExamples:")
    print("  python run_server.py                # รัน server")
    print("  python run_server.py --check        # ตรวจสอบ environment")
    print("  python run_server.py --setup        # รัน setup")

def run_setup():
    """รัน setup script"""
    print("🔧 Running setup script...")
    
    try:
        result = subprocess.run([sys.executable, "setup_backend.py"], check=True)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"❌ Setup failed: {e}")
        return False
    except FileNotFoundError:
        print("❌ setup_backend.py not found")
        return False

def main():
    """Main function"""
    
    # ตรวจสอบ command line arguments
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        
        if arg in ["--help", "-h"]:
            show_usage()
            return 0
        elif arg in ["--setup", "-s"]:
            return 0 if run_setup() else 1
        elif arg in ["--check", "-c"]:
            print_banner()
            if check_environment() and check_dependencies():
                print("\n✅ Environment check passed!")
                return 0
            else:
                print("\n❌ Environment check failed!")
                return 1
        else:
            print(f"Unknown option: {arg}")
            show_usage()
            return 1
    
    # รัน server (default behavior)
    print_banner()
    
    # ตรวจสอบ environment
    if not check_environment():
        print("\n💡 Tip: รัน 'python run_server.py --setup' เพื่อติดตั้งระบบ")
        return 1
    
    # ตรวจสอบ dependencies
    if not check_dependencies():
        print("\n💡 Tip: รัน 'pip install -r requirements.txt' เพื่อติดตั้ง dependencies")
        return 1
    
    # สร้าง directories ที่จำเป็น
    if not create_required_directories():
        return 1
    
    # เริ่มต้น server
    if not start_server():
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)