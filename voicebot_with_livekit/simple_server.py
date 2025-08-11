#!/usr/bin/env python3
"""
Simple HTTP server to serve the index.html file
"""
import http.server
import socketserver
import webbrowser
from pathlib import Path

def start_simple_server(port=8000):
    """Start a simple HTTP server"""
    try:
        with socketserver.TCPServer(("", port), http.server.SimpleHTTPRequestHandler) as httpd:
            print(f"🌐 Simple HTTP Server running at: http://localhost:{port}")
            print(f"📄 Open: http://localhost:{port}/index.html")
            print("Press Ctrl+C to stop")
            
            # Try to open browser
            try:
                webbrowser.open(f'http://localhost:{port}/index.html')
                print("🌐 Opening browser automatically...")
            except:
                print("💡 Please open the URL manually in your browser")
            
            httpd.serve_forever()
            
    except KeyboardInterrupt:
        print("\n👋 Server stopped")
    except OSError as e:
        if "Address already in use" in str(e):
            print(f"❌ Port {port} is already in use. Try: python simple_server.py")
        else:
            print(f"❌ Server error: {e}")

if __name__ == "__main__":
    start_simple_server()