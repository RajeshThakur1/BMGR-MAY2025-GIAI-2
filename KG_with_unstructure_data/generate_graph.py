#!/usr/bin/env python3
"""
Knowledge Graph Generator and Visualizer.
This script is the entry point for generating knowledge graphs from textual data.
"""
import sys
import os

# Add the current directory to the Python path to find the module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.knowledge_graph.main import main

if __name__ == "__main__":
    main() 