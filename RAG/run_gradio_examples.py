#!/usr/bin/env python3
"""
Interactive Gradio Tutorial Runner
Run this to execute Gradio examples step by step
"""

import os
import sys

def main():
    print("🎯 Gradio Step-by-Step Tutorial")
    print("=" * 50)
    
    while True:
        print("\nChoose an example to run:")
        print("1. Simple Greeting App")
        print("2. BMI Calculator (Multiple inputs/outputs)")
        print("3. Text Analyzer (Dropdown, Slider, JSON)")
        print("4. Image Filter (Image processing)")
        print("5. Function Plotter (Data visualization)")
        print("6. Custom Calculator (Gradio Blocks)")
        print("7. Simple Chatbot")
        print("8. Mini RAG System")
        print("0. Exit")
        
        try:
            choice = input("\nEnter your choice (0-8): ").strip()
            
            if choice == "0":
                print("👋 Thanks for learning Gradio!")
                break
            elif choice in ["1", "2", "3", "4", "5", "6", "7", "8"]:
                print(f"\n🚀 Loading example {choice}...")
                # Import and run the example
                exec(f"from gradio_step_by_step import launch_demo; launch_demo({choice})")
            else:
                print("❌ Invalid choice. Please enter a number 0-8.")
                
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            print("Make sure gradio_step_by_step.py is in the same directory.")

if __name__ == "__main__":
    main() 