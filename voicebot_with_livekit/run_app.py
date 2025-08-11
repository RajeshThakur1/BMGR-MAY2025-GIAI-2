#!/usr/bin/env python3
"""
All-in-one script to run both the web server and voice agent
"""
import subprocess
import sys
import time
import os
import signal
import webbrowser
from pathlib import Path

def run_voice_bot():
    """Run the complete voice bot application"""
    print("🤖 Starting Voice Bot Application...")
    print("=" * 50)
    
    # Check if we're in a virtual environment
    if not hasattr(sys, 'real_prefix') and not (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("⚠️  Warning: Virtual environment not detected!")
        print("💡 Please activate your virtual environment first:")
        print("   source venv/bin/activate")
        print("   python run_app.py")
        return
    
    processes = []
    
    try:
        # Start token server
        print("🔑 Starting token server...")
        token_process = subprocess.Popen([
            sys.executable, "token_server.py"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        processes.append(token_process)
        
        # Wait a moment for token server to start
        time.sleep(1)
        
        # Start web server
        print("🌐 Starting web server...")
        server_process = subprocess.Popen([
            sys.executable, "server.py"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        processes.append(server_process)
        
        # Wait a moment for server to start
        time.sleep(2)
        
        # Start voice agent
        print("🎤 Starting voice agent...")
        agent_process = subprocess.Popen([
            sys.executable, "web_agent.py", "dev"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        processes.append(agent_process)
        
        # Wait a moment for agent to start
        time.sleep(3)
        
        print("\n✅ All services started successfully!")
        print("🎯 Simple Interface: http://localhost:8080/simple_voice.html")
        print("🔧 Working Interface: http://localhost:8080/working_app.html")
        print("📱 Live Interface: http://localhost:8080/live_app.html")
        print("🧪 Demo Interface: http://localhost:8080/app.html")
        print("🔑 Token Server: http://localhost:3001")
        print("\n💡 Instructions:")
        print("1. Open http://localhost:8080/simple_voice.html in your browser")
        print("2. Wait for SDK to load (check status)")
        print("3. Enter room name: 'voice-chat-room'")
        print("4. Enter your name")
        print("5. Click 'Connect to Voice Bot'")
        print("6. Allow microphone access when prompted")
        print("7. Click and hold the microphone to speak to the AI")
        print("\nPress Ctrl+C to stop all services")
        
        # Try to open browser
        try:
            webbrowser.open('http://localhost:8080/simple_voice.html')
            print("🌐 Opening simple interface automatically...")
        except:
            pass
        
        # Monitor processes
        while True:
            time.sleep(1)
            
            # Check if processes are still running
            for i, process in enumerate(processes):
                if process.poll() is not None:
                    print(f"❌ Process {i+1} stopped unexpectedly")
                    stdout, stderr = process.communicate()
                    if stdout:
                        print(f"Output: {stdout}")
                    if stderr:
                        print(f"Error: {stderr}")
    
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        
    finally:
        # Clean up processes
        for process in processes:
            try:
                process.terminate()
                process.wait(timeout=5)
            except:
                try:
                    process.kill()
                except:
                    pass
        print("👋 All services stopped")

if __name__ == "__main__":
    run_voice_bot()