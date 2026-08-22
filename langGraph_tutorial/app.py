from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END, START
from dotenv import load_dotenv

import os
import base64
from pathlib import Path
import gradio as gr
from io import BytesIO
from PIL import Image
from openai import OpenAI
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

APP_DIR = Path(__file__).resolve().parent
IMAGES_DIR = APP_DIR / "images"
IMAGES_DIR.mkdir(exist_ok=True)

class State(TypedDict):
    product_name: str
    basic_description: str
    features_benefits: str
    marketing_message: str
    image_url: str
    final_description: str


def image_path_for(product_name: str) -> Path:
    safe_name = "".join(ch if ch.isalnum() else "_" for ch in product_name.lower())
    safe_name = "_".join(part for part in safe_name.split("_") if part)
    return IMAGES_DIR / f"{safe_name}.png"


def generate_basic_description(state: State) -> State:
    product_name = state["product_name"]
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that generates product descriptions.",
            },
            {
                "role": "user",
                "content": f"Generate a basic description for the product: {product_name}.",
            },
        ],
    )
    return {"basic_description": response.choices[0].message.content.strip()}


def add_features_benefits(state: State) -> State:
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that adds features and benefits to product descriptions.",
            },
            {
                "role": "user",
                "content": (
                    f"Based on the basic description: {state['basic_description']}, "
                    "list the features and benefits of the product."
                ),
            },
        ],
    )
    return {"features_benefits": response.choices[0].message.content.strip()}

def create_image(state: State) -> State:
    prompt = (
        f"Professional product photo of {state['product_name']}. "
        f"Clean background, marketing style. "
        f"Features: {state['features_benefits'][:500]}"
    )
    response = client.images.generate(
        model="gpt-image-2",
        prompt=prompt,
        size="1024x1024",
        n=1,
    )
    image_b64 = response.data[0].b64_json
    image_path_for(state["product_name"]).write_bytes(base64.b64decode(image_b64))
    return {"image_url": f"data:image/png;base64,{image_b64}"}


def create_marketing_message(state: State) -> State:
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that creates marketing messages for products.",
            },
            {
                "role": "user",
                "content": (
                    "Create a marketing message based on the features and benefits: "
                    f"{state['features_benefits']}."
                ),
            },
        ],
    )
    return {"marketing_message": response.choices[0].message.content.strip()}

def polish_final_description(state: State) -> State:
    user_content = [
        {
            "type": "text",
            "text": (
                "Polish the following into one compelling final product description. "
                "Reference what you see in the product image (look, design, key visual features).\n\n"
                f"Basic Description: {state['basic_description']}\n"
                f"Marketing Message: {state['marketing_message']}"
            ),
        },
        {
            "type": "image_url",
            "image_url": {"url": state["image_url"]},
        },
    ]
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that polishes product descriptions using text and images.",
            },
            {"role": "user", "content": user_content},
        ],
    )
    return {"final_description": response.choices[0].message.content.strip()}


def build_workflow():
    workflow = StateGraph(State)
    workflow.add_node("generate_basic_description", generate_basic_description)
    workflow.add_node("add_features_benefits", add_features_benefits)
    workflow.add_node("create_image", create_image)
    workflow.add_node("create_marketing_message", create_marketing_message)
    workflow.add_node("polish_final_description", polish_final_description)

    workflow.add_edge(START, "generate_basic_description")
    workflow.add_edge("generate_basic_description", "add_features_benefits")
    workflow.add_edge("add_features_benefits", "create_image")
    workflow.add_edge("create_image", "create_marketing_message")
    workflow.add_edge("create_marketing_message", "polish_final_description")
    workflow.add_edge("polish_final_description", END)
    return workflow.compile()


chain = build_workflow()

def generate_product(product_name: str):
    product_name = (product_name or "").strip()
    if not product_name:
        raise gr.Error("Enter a product name.")

    state = State(
        product_name=product_name,
        basic_description="",
        features_benefits="",
        marketing_message="",
        image_url="",
        final_description="",
    )
    basic = features = marketing = final = ""
    image = None
    status = "Starting LangGraph workflow..."

    def snapshot():
        return status, image if image is not None else gr.skip(), final, basic, features, marketing


    try:
        for event in chain.stream(state):
            _node_name, updates = next(iter(event.items()))
            if "basic_description" in updates:
                basic = updates["basic_description"]
                status = "Generated basic description"
            if "features_benefits" in updates:
                features = updates["features_benefits"]
                status = "Listed features and benefits"
                yield snapshot()
                status = "Creating product image..."
                yield snapshot()
                continue
            if "image_url" in updates:
                image_url = updates["image_url"]
                if image_url.startswith("data:image"):
                    image_b64 = image_url.split(",", 1)[1]
                    image = Image.open(BytesIO(base64.b64decode(image_b64))).convert("RGB")
                status = "Created product image"
            if "marketing_message" in updates:
                marketing = updates["marketing_message"]
                status = "Wrote marketing message"
            if "final_description" in updates:
                final = updates["final_description"]
                status = "Polished final description"
            yield snapshot()
        yield "Done", image if image is not None else gr.skip(), final, basic, features, marketing
    except Exception as exc:
        raise gr.Error(str(exc)) from exc


with gr.Blocks(title="LangGraph Product Studio") as demo:
    gr.Markdown(
        """
        # LangGraph Product Studio
        Enter a product name. The graph runs:
        **description → features → image → marketing → polished copy**.
        """
    )
    with gr.Row():
        product_name = gr.Textbox(
            label="Product name",
            value="Smart Fitness Watch",
            scale=3,
        )
        generate_btn = gr.Button("Generate", variant="primary", scale=1)
    status = gr.Textbox(label="Pipeline status", interactive=False)
    image = gr.Image(
        label="Generated product image",
        type="pil",
        format="png",
        interactive=True,
        height=420,
    )
    final = gr.Textbox(label="Final description", lines=10)
    with gr.Accordion("Intermediate graph outputs", open=False):
        basic = gr.Textbox(label="1. Basic description", lines=6)
        features = gr.Textbox(label="2. Features and benefits", lines=8)
        marketing = gr.Textbox(label="3. Marketing message", lines=6)
    gr.Examples(
        examples=["Smart Fitness Watch", "Wireless Earbuds", "Stainless Steel Water Bottle"],
        inputs=product_name,
    )
    generate_btn.click(
        fn=generate_product,
        inputs=product_name,
        outputs=[status, image, final, basic, features, marketing],
    )
    product_name.submit(
        fn=generate_product,
        inputs=product_name,
        outputs=[status, image, final, basic, features, marketing],
    )


if __name__ == "__main__":
    demo.launch(allowed_paths=[str(IMAGES_DIR)])
