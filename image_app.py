import gradio as gr
import torch
import diffusers
from PIL import Image
import os
import random
from datetime import datetime

# ==============================================================================
# ULTIMATE GPU DIAGNOSTIC
# ==============================================================================
print("="*60)
print("              INITIALIZING: GPU & PYTORCH DIAGNOSTIC")
print("="*60)
print(f"PyTorch Version: {torch.__version__}")
IS_CUDA_AVAILABLE = torch.cuda.is_available()
print(f"Is CUDA available? ---> {IS_CUDA_AVAILABLE}")

if IS_CUDA_AVAILABLE:
    print(f"CUDA Version (linked to PyTorch): {torch.version.cuda}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  - GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    print("\nWARNING: PyTorch cannot find a CUDA-enabled GPU.")
    print("The application will fall back to CPU, which will be extremely slow.")
print("="*60)
# ==============================================================================


# --- SDNQ Integration ---
try:
    from sdnq import SDNQConfig
    from sdnq.common import use_torch_compile as triton_is_available
    from sdnq.loader import apply_sdnq_options_to_model
    print("SDNQ library loaded successfully.")
except ImportError:
    print("FATAL ERROR: The 'sdnq' library is not installed.")
    raise

# --- Model Caching & Configuration ---
cache_dir = os.path.join(os.getcwd(), "Models")
output_dir = os.path.join(os.getcwd(), "output")
os.makedirs(cache_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True) # Create output directory
os.environ['HF_HOME'] = cache_dir
os.environ['HUGGINGFACE_HUB_CACHE'] = cache_dir
print(f"Models will be downloaded to: {cache_dir}")
print(f"Images will be saved to: {output_dir}")

MODELS = {
    "Z-Image-Turbo-SDNQ-uint4-svd-r32": {"type": "text-to-image", "id": "Disty0/Z-Image-Turbo-SDNQ-uint4-svd-r32", "pipeline": diffusers.ZImagePipeline, "instance": None, "defaults": {"steps": 9, "guidance": 0.0}},
    "Qwen-Image-2512-SDNQ-4bit-dynamic": {"type": "text-to-image", "id": "Disty0/Qwen-Image-2512-SDNQ-4bit-dynamic", "pipeline": diffusers.AutoPipelineForText2Image, "instance": None, "defaults": {"steps": 50, "guidance": 4.0}},
    "FLUX.2-klein-4B-SDNQ-4bit-dynamic": {"type": "image-editing", "id": "Disty0/FLUX.2-klein-4B-SDNQ-4bit-dynamic", "pipeline": diffusers.AutoPipelineForImage2Image, "instance": None, "defaults": {"steps": 4, "guidance": 1.0}},
    "FLUX.2-klein-9B-SDNQ-4bit-dynamic-svd-r32": {"type": "image-editing", "id": "Disty0/FLUX.2-klein-9B-SDNQ-4bit-dynamic-svd-r32", "pipeline": diffusers.AutoPipelineForImage2Image, "instance": None, "defaults": {"steps": 4, "guidance": 1.0}},
    "Qwen-Image-Edit-2511-SDNQ-uint4-svd-r32": {"type": "image-editing", "id": "Disty0/Qwen-Image-Edit-2511-SDNQ-uint4-svd-r32", "pipeline": diffusers.AutoPipelineForImage2Image, "instance": None, "defaults": {"steps": 40, "guidance": 4.0}}
}

ASPECT_RATIOS = {"1:1 (Square)": (1024, 1024), "16:9 (Landscape)": (1344, 768), "9:16 (Portrait)": (768, 1344), "4:3 (Standard)": (1024, 768), "3:4 (Standard Portrait)": (768, 1024)}

# --- Core Functions ---
def get_model_names(model_type):
    return [name for name, config in MODELS.items() if config["type"] == model_type]

def load_model(model_name):
    if model_name not in MODELS: raise gr.Error(f"Model '{model_name}' not found.")
    model_info = MODELS[model_name]
    if model_info["instance"] is None:
        print(f"Loading {model_name} with SDNQ...")
        try:
            pipe = model_info["pipeline"].from_pretrained(model_info["id"], torch_dtype=torch.bfloat16, cache_dir=cache_dir)
            if triton_is_available and IS_CUDA_AVAILABLE:
                print("Applying SDNQ MatMul optimizations...")
                if hasattr(pipe, 'transformer'): pipe.transformer = apply_sdnq_options_to_model(pipe.transformer, use_quantized_matmul=True)
                if hasattr(pipe, 'text_encoder'): pipe.text_encoder = apply_sdnq_options_to_model(pipe.text_encoder, use_quantized_matmul=True)
            if IS_CUDA_AVAILABLE:
                print("CUDA detected. Enabling model CPU offload for VRAM savings.")
                pipe.enable_model_cpu_offload()
            else:
                pipe.to("cpu")
            model_info["instance"] = pipe
            print(f"{model_name} loaded successfully.")
        except Exception as e:
            raise gr.Error(f"Failed to load model {model_name}. Error: {e}")
    return model_info["instance"]

def save_image(image):
    """Saves a PIL image to the output directory with a unique name."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"{timestamp}.png"
    filepath = os.path.join(output_dir, filename)
    image.save(filepath)
    print(f"Image saved to {filepath}")

def generate_image_from_text(model_name, prompt, negative_prompt, steps, guidance, width, height, seed, randomize_seed):
    if not prompt: raise gr.Error("Prompt cannot be empty.")
    
    if randomize_seed:
        seed = random.randint(0, 2**32 - 1)
    
    pipe = load_model(model_name)
    image = pipe(prompt=prompt, negative_prompt=negative_prompt, num_inference_steps=int(steps), guidance_scale=guidance, width=int(width), height=int(height), generator=torch.manual_seed(int(seed))).images[0]
    
    save_image(image)
    return image, gr.update(value=seed)

def edit_images(model_name, prompt, input_images, steps, guidance, seed, randomize_seed):
    if not prompt or not input_images: raise gr.Error("Prompt and at least one input image are required.")
    
    pipe = load_model(model_name)
    output_images = []
    
    for img_tuple in input_images:
        if randomize_seed:
            current_seed = random.randint(0, 2**32 - 1)
        else:
            current_seed = int(seed)
            
        generator = torch.manual_seed(current_seed)
        pil_image = Image.fromarray(img_tuple[0])
        
        # All editing models are instruct-based and do not use 'strength'
        call_args = {
            "prompt": prompt, "image": pil_image,
            "num_inference_steps": int(steps), "guidance_scale": guidance,
            "generator": generator
        }

        edited_image = pipe(**call_args).images[0]
        save_image(edited_image)
        output_images.append(edited_image)
        
    new_seed_value = current_seed if randomize_seed else seed
    return output_images, gr.update(value=new_seed_value)

# --- UI Helper Functions ---
def update_sliders(model_name):
    defaults = MODELS[model_name]["defaults"]
    return gr.update(value=defaults["steps"]), gr.update(value=defaults["guidance"])

def update_resolution(aspect_ratio):
    width, height = ASPECT_RATIOS[aspect_ratio]
    return gr.update(value=width), gr.update(value=height)

# --- Gradio UI ---
custom_theme = gr.themes.Monochrome(primary_hue="orange", secondary_hue="zinc", neutral_hue="stone").set(button_primary_background_fill_dark="*primary_500", button_primary_background_fill_hover_dark="*primary_600")

with gr.Blocks(title="Studio X") as demo: 
    gr.Image("logo.gif", show_label=False, container=False)
    gr.Markdown("<h2 style='text-align: center;'>A Local AI Interface, SDNQ Quantized Models</h2>")
    
    with gr.Row(elem_id="seed_row"):
        seed = gr.Number(label="Seed", value=42, precision=0)
        randomize_seed = gr.Checkbox(label="Randomize Seed", value=False)

    with gr.Tabs():
        with gr.TabItem("Text-to-Image"):
            t2i_model_names = get_model_names("text-to-image")
            with gr.Row():
                with gr.Column(scale=2):
                    t2i_model_name = gr.Dropdown(label="Select Model", choices=t2i_model_names, value=t2i_model_names[0])
                    prompt = gr.Textbox(label="Prompt", placeholder="A stunning fantasy landscape", lines=3)
                    negative_prompt = gr.Textbox(label="Negative Prompt", placeholder="Blurry, low quality, ugly")
                    with gr.Accordion("Resolution", open=True):
                        aspect_ratio = gr.Dropdown(label="Aspect Ratio", choices=list(ASPECT_RATIOS.keys()), value="1:1 (Square)")
                        t2i_width = gr.Slider(label="Width", minimum=256, maximum=2048, value=1024, step=64)
                        t2i_height = gr.Slider(label="Height", minimum=256, maximum=2048, value=1024, step=64)
                    with gr.Accordion("Generation Parameters", open=True):
                        t2i_steps = gr.Slider(label="Inference Steps", minimum=1, maximum=100, step=1)
                        t2i_guidance = gr.Slider(label="Guidance Scale", minimum=0.0, maximum=20.0, step=0.1)
                    t2i_button = gr.Button("Generate", variant="primary")
                with gr.Column(scale=1):
                    t2i_output_image = gr.Image(label="Generated Image", type="pil")

        with gr.TabItem("Image Editing"):
            i2i_model_names = get_model_names("image-editing")
            with gr.Row():
                with gr.Column(scale=2):
                    i2i_model_name = gr.Dropdown(label="Select Model", choices=i2i_model_names, value=i2i_model_names[0])
                    edit_prompt = gr.Textbox(label="Edit Instruction", placeholder="Make it a watercolor painting", lines=3)
                    input_images = gr.Gallery(label="Upload Images", type="numpy", show_label=True)
                    with gr.Accordion("Generation Parameters", open=True):
                        # REMOVED: The strength slider is gone
                        i2i_steps = gr.Slider(label="Inference Steps", minimum=1, maximum=100, step=1)
                        i2i_guidance = gr.Slider(label="Guidance Scale (CFG)", minimum=0.0, maximum=20.0, step=0.1)
                    i2i_button = gr.Button("Edit Image(s)", variant="primary")
                with gr.Column(scale=1):
                    i2i_output_images = gr.Gallery(label="Edited Images", columns=2, show_label=True)

    # --- Event Handlers ---
    t2i_model_name.change(fn=update_sliders, inputs=t2i_model_name, outputs=[t2i_steps, t2i_guidance])
    i2i_model_name.change(fn=update_sliders, inputs=i2i_model_name, outputs=[i2i_steps, i2i_guidance])
    aspect_ratio.change(fn=update_resolution, inputs=aspect_ratio, outputs=[t2i_width, t2i_height])

    demo.load(fn=update_sliders, inputs=t2i_model_name, outputs=[t2i_steps, t2i_guidance])
    demo.load(fn=update_sliders, inputs=i2i_model_name, outputs=[i2i_steps, i2i_guidance])

    t2i_button.click(
        fn=generate_image_from_text,
        inputs=[t2i_model_name, prompt, negative_prompt, t2i_steps, t2i_guidance, t2i_width, t2i_height, seed, randomize_seed],
        outputs=[t2i_output_image, seed]
    )
    i2i_button.click(
        fn=edit_images,
        # REMOVED: strength is no longer an input
        inputs=[i2i_model_name, edit_prompt, input_images, i2i_steps, i2i_guidance, seed, randomize_seed],
        outputs=[i2i_output_images, seed]
    )

if __name__ == "__main__":
    demo.launch(theme=custom_theme)
