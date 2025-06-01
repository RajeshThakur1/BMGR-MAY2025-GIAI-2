# Gradio Step-by-Step Tutorial
# Execute each section one by one to learn Gradio progressively

# ============================================================================
# STEP 1: Installation and Imports
# ============================================================================

# First, install required packages (run this in terminal):
# pip install gradio pandas numpy matplotlib pillow

import gradio as gr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io

print("✅ All imports successful!")

# ============================================================================
# STEP 2: Your First Gradio Interface
# ============================================================================

def greet(name):
    """Simple greeting function"""
    return f"Hello, {name}! Welcome to Gradio."

# Create and launch your first interface
demo1 = gr.Interface(
    fn=greet,
    inputs="text",
    outputs="text",
    title="My First Gradio App"
)

# Run this to launch the interface
# demo1.launch()

print("✅ First interface created! Uncomment demo1.launch() to run it.")

# ============================================================================
# STEP 3: Multiple Inputs and Outputs
# ============================================================================

def calculate_bmi(weight, height):
    """Calculate BMI and return classification"""
    bmi = weight / (height ** 2)
    
    if bmi < 18.5:
        category = "Underweight"
    elif bmi < 25:
        category = "Normal weight"
    elif bmi < 30:
        category = "Overweight"
    else:
        category = "Obese"
    
    return round(bmi, 2), category

# Create interface with multiple inputs and outputs
demo2 = gr.Interface(
    fn=calculate_bmi,
    inputs=[
        gr.Number(label="Weight (kg)", value=70),
        gr.Number(label="Height (m)", value=1.75)
    ],
    outputs=[
        gr.Number(label="BMI"),
        gr.Text(label="Category")
    ],
    title="BMI Calculator"
)

# Run this to test
# demo2.launch()

print("✅ BMI Calculator created!")

# ============================================================================
# STEP 4: Working with Different Input Types
# ============================================================================

def analyze_text(text, case_option, word_limit):
    """Analyze and transform text based on options"""
    if not text:
        return "Please enter some text", 0, []
    
    # Apply case transformation
    if case_option == "Upper":
        transformed = text.upper()
    elif case_option == "Lower":
        transformed = text.lower()
    elif case_option == "Title":
        transformed = text.title()
    else:
        transformed = text
    
    # Count words
    words = text.split()
    word_count = len(words)
    
    # Get limited words
    limited_words = words[:word_limit] if word_limit > 0 else words
    
    return transformed, word_count, limited_words

demo3 = gr.Interface(
    fn=analyze_text,
    inputs=[
        gr.Textbox(label="Input Text", lines=3, placeholder="Enter your text here..."),
        gr.Dropdown(["Original", "Upper", "Lower", "Title"], label="Case Transform"),
        gr.Slider(1, 20, value=10, label="Word Limit")
    ],
    outputs=[
        gr.Textbox(label="Transformed Text"),
        gr.Number(label="Word Count"),
        gr.JSON(label="Limited Words")
    ],
    title="Text Analyzer"
)

print("✅ Text Analyzer created!")

# ============================================================================
# STEP 5: Image Processing
# ============================================================================

def simple_image_filter(image, filter_type):
    """Apply simple filters to images"""
    if image is None:
        return None
    
    # Convert to numpy array
    img_array = np.array(image)
    
    if filter_type == "Grayscale":
        # Convert to grayscale
        gray = np.dot(img_array[...,:3], [0.299, 0.587, 0.114])
        return np.stack([gray, gray, gray], axis=-1).astype(np.uint8)
    
    elif filter_type == "Red Channel":
        # Keep only red channel
        red_only = img_array.copy()
        red_only[:,:,1] = 0  # Remove green
        red_only[:,:,2] = 0  # Remove blue
        return red_only
    
    elif filter_type == "Invert":
        # Invert colors
        return 255 - img_array
    
    else:  # Original
        return img_array

demo4 = gr.Interface(
    fn=simple_image_filter,
    inputs=[
        gr.Image(type="pil", label="Upload Image"),
        gr.Radio(["Original", "Grayscale", "Red Channel", "Invert"], 
                value="Original", label="Filter")
    ],
    outputs=gr.Image(label="Filtered Image"),
    title="Simple Image Filter"
)

print("✅ Image Filter created!")

# ============================================================================
# STEP 6: Data Visualization
# ============================================================================

def create_simple_plot(plot_type, num_points):
    """Create different types of plots"""
    x = np.linspace(0, 10, num_points)
    
    if plot_type == "Sine":
        y = np.sin(x)
        title = "Sine Wave"
    elif plot_type == "Cosine":
        y = np.cos(x)
        title = "Cosine Wave"
    elif plot_type == "Linear":
        y = 2 * x + 1
        title = "Linear Function"
    else:  # Quadratic
        y = x ** 2
        title = "Quadratic Function"
    
    plt.figure(figsize=(8, 5))
    plt.plot(x, y, 'b-', linewidth=2)
    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True, alpha=0.3)
    
    return plt.gcf()

demo5 = gr.Interface(
    fn=create_simple_plot,
    inputs=[
        gr.Dropdown(["Sine", "Cosine", "Linear", "Quadratic"], 
                   value="Sine", label="Function Type"),
        gr.Slider(10, 100, value=50, step=10, label="Number of Points")
    ],
    outputs=gr.Plot(label="Generated Plot"),
    title="Function Plotter"
)

print("✅ Function Plotter created!")

# ============================================================================
# STEP 7: Using Gradio Blocks for Custom Layout
# ============================================================================

def add_numbers(a, b):
    return a + b

def multiply_numbers(a, b):
    return a * b

# Create custom interface using Blocks
with gr.Blocks(title="Calculator") as demo6:
    gr.Markdown("# Simple Calculator")
    
    with gr.Row():
        num1 = gr.Number(label="First Number", value=0)
        num2 = gr.Number(label="Second Number", value=0)
    
    with gr.Row():
        add_btn = gr.Button("Add", variant="primary")
        multiply_btn = gr.Button("Multiply", variant="secondary")
    
    result = gr.Number(label="Result")
    
    # Connect buttons to functions
    add_btn.click(add_numbers, inputs=[num1, num2], outputs=result)
    multiply_btn.click(multiply_numbers, inputs=[num1, num2], outputs=result)

print("✅ Custom Calculator created!")

# ============================================================================
# STEP 8: Chatbot Interface
# ============================================================================

def simple_chatbot(message, history):
    """Simple echo chatbot that responds to user messages"""
    # Simple responses based on keywords
    message_lower = message.lower()
    
    if "hello" in message_lower or "hi" in message_lower:
        response = "Hello! How can I help you today?"
    elif "how are you" in message_lower:
        response = "I'm doing great! Thanks for asking."
    elif "weather" in message_lower:
        response = "I don't have access to weather data, but I hope it's nice where you are!"
    elif "python" in message_lower:
        response = "Python is a great programming language! Are you learning it?"
    elif "gradio" in message_lower:
        response = "Gradio is awesome for creating ML interfaces quickly!"
    elif "bye" in message_lower or "goodbye" in message_lower:
        response = "Goodbye! Have a great day!"
    else:
        response = f"You said: '{message}'. That's interesting! Tell me more."
    
    # Add to history
    history.append([message, response])
    return history, ""

# Create chatbot interface
with gr.Blocks() as demo7:
    gr.Markdown("# Simple Chatbot")
    
    chatbot = gr.Chatbot(height=300)
    msg = gr.Textbox(label="Type your message", placeholder="Say hello!")
    send = gr.Button("Send")
    clear = gr.Button("Clear")
    
    # Connect events
    send.click(simple_chatbot, inputs=[msg, chatbot], outputs=[chatbot, msg])
    msg.submit(simple_chatbot, inputs=[msg, chatbot], outputs=[chatbot, msg])
    clear.click(lambda: ([], ""), outputs=[chatbot, msg])

print("✅ Simple Chatbot created!")

# ============================================================================
# STEP 9: Mini RAG System
# ============================================================================

# Simple knowledge base
knowledge_base = [
    "Python is a high-level programming language known for its simplicity.",
    "Machine learning is a subset of AI that learns from data.",
    "Gradio helps create user interfaces for machine learning models.",
    "Neural networks are inspired by the human brain structure.",
    "Data science combines statistics, programming, and domain expertise."
]

def simple_rag(question, top_k):
    """Simple RAG system that finds relevant info and responds"""
    question_lower = question.lower()
    
    # Simple keyword matching
    relevant_docs = []
    for doc in knowledge_base:
        score = 0
        for word in question_lower.split():
            if word in doc.lower():
                score += 1
        if score > 0:
            relevant_docs.append((doc, score))
    
    # Sort by relevance and take top_k
    relevant_docs.sort(key=lambda x: x[1], reverse=True)
    top_docs = [doc for doc, _ in relevant_docs[:top_k]]
    
    if not top_docs:
        return "I don't have information about that topic.", "No relevant documents found."
    
    # Create response
    response = f"Based on my knowledge:\n\n"
    for i, doc in enumerate(top_docs, 1):
        response += f"{i}. {doc}\n"
    
    context = "\n".join([f"• {doc}" for doc in top_docs])
    
    return response, context

demo8 = gr.Interface(
    fn=simple_rag,
    inputs=[
        gr.Textbox(label="Ask a question", 
                  placeholder="Ask about Python, ML, Gradio, etc."),
        gr.Slider(1, 3, value=2, step=1, label="Number of sources")
    ],
    outputs=[
        gr.Textbox(label="Answer", lines=5),
        gr.Textbox(label="Sources Used", lines=4)
    ],
    title="Mini RAG System",
    examples=[
        ["What is Python?", 2],
        ["Tell me about machine learning", 2],
        ["How does Gradio work?", 1]
    ]
)

print("✅ Mini RAG System created!")

# ============================================================================
# STEP 10: Launch Functions (Use these to test each demo)
# ============================================================================

def launch_demo(demo_number):
    """Helper function to launch specific demos"""
    demos = {
        1: demo1,  # Simple greeting
        2: demo2,  # BMI Calculator
        3: demo3,  # Text Analyzer
        4: demo4,  # Image Filter
        5: demo5,  # Function Plotter
        6: demo6,  # Custom Calculator
        7: demo7,  # Simple Chatbot
        8: demo8,  # Mini RAG System
    }
    
    if demo_number in demos:
        print(f"Launching Demo {demo_number}...")
        demos[demo_number].launch(share=False, inbrowser=True)
    else:
        print("Demo number not found!")

# ============================================================================
# HOW TO USE THIS FILE:
# ============================================================================

print("\n" + "="*60)
print("🎉 GRADIO TUTORIAL READY!")
print("="*60)
print("\nTo run each demo, use:")
print("launch_demo(1)  # Simple greeting")
print("launch_demo(2)  # BMI Calculator") 
print("launch_demo(3)  # Text Analyzer")
print("launch_demo(4)  # Image Filter")
print("launch_demo(5)  # Function Plotter")
print("launch_demo(6)  # Custom Calculator")
print("launch_demo(7)  # Simple Chatbot")
print("launch_demo(8)  # Mini RAG System")
print("\nExample:")
print(">>> launch_demo(1)")
print("\n" + "="*60)

# Test with a simple example
if __name__ == "__main__":
    print("\n🚀 Testing simple function:")
    result = greet("World")
    print(f"greet('World') = '{result}'")
    
    print("\n🧮 Testing BMI calculation:")
    bmi, category = calculate_bmi(70, 1.75)
    print(f"BMI for 70kg, 1.75m = {bmi} ({category})") 