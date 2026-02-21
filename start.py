#!/usr/bin/env python3
"""
One-click launcher for Distributed ML Platform
"""

import subprocess
import sys
import os
import time
import socket
import webbrowser
import signal

# Store processes
processes = []

def cleanup(signum=None, frame=None):
    print("\n👋 Shutting down...")
    for p in processes:
        p.terminate()
    sys.exit(0)

signal.signal(signal.SIGINT, cleanup)

def main():
    print("\n" + "="*60)
    print("🚀 STARTING DISTRIBUTED ML PLATFORM")
    print("="*60)
    
    # Get IP
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
        s.close()
    except:
        ip = "localhost"
    
    # Start coordinator
    print("\n📡 Starting Coordinator...")
    coord = subprocess.Popen([sys.executable, "backend/coordinator.py"])
    processes.append(coord)
    time.sleep(2)
    
    # Start dashboard
    print("📊 Starting Dashboard...")
    dash = subprocess.Popen([sys.executable, "frontend/app.py"])
    processes.append(dash)
    time.sleep(2)
    
    print("\n" + "="*60)
    print("✅ SYSTEM RUNNING!")
    print("="*60)
    print(f"📊 Dashboard: http://{ip}:5000")
    print(f"🔌 Coordinator: {ip}:50051")
    print("\n📝 To add a worker:")
    print(f"   python backend/worker.py --coordinator {ip}")
    print("\n⚠️  Press Ctrl+C to stop")
    print("="*60 + "\n")
    
    # Open browser
    webbrowser.open(f"http://{ip}:5000")
    
    # Wait
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        cleanup()

if __name__ == "__main__":
    main()