<div align="center">

# Studio X

### A Simplified Local AI Interface for Image Generation & Editing

![Studio X Logo](logo.gif)

</div>

---

**Studio X** is a user-friendly desktop application for running powerful, locally-hosted AI image models. Built with Python and Gradio, it provides a simple interface to perform text-to-image generation and instruct-based image editing without needing to write any code.

This project is specifically designed to leverage highly efficient **SDNQ quantized models**, allowing for faster performance and lower VRAM usage on consumer hardware.

## ✨ Features

- **Dual-Mode Generation:** Full support for both **Text-to-Image** and **Image Editing** in separate, organized tabs.
- **Optimized for Local Use:** Runs entirely on your own machine. Your data stays private.
- **Simple Setup:** Includes a batch script (`run.bat`) that automatically creates a virtual environment and installs all necessary dependencies.
- **Multi-Model Support:** Easily switch between different text-to-image and image-editing models via a dropdown menu.
- **Smart Settings:** Automatically suggests the optimal inference steps and guidance scale for each selected model.
- **Full Resolution Control:** Choose from common aspect ratio presets or manually adjust width and height with sliders.
- **Reproducibility & Creativity:**
    - Manually set a **Seed** for reproducible results.
    - Use the **"Randomize Seed"** checkbox for endless creative variations.
- **Automatic File Saving:** Every generated image is automatically saved to an `output` folder for your convenience.
- **VRAM Efficiency:** Intelligently enables CPU offloading on systems with a detected CUDA GPU to run large models with less VRAM.

## 🤖 Supported Models

Studio X comes pre-configured to support the following quantized models from [Disty0 on Hugging Face](https://huggingface.co/Disty0):

| Model Name                               | Type             | Use Case                                  |
| ---------------------------------------- | ---------------- | ----------------------------------------- |
| **Z-Image-Turbo-SDNQ-uint4-svd-r32**       | Text-to-Image    | High-speed, quality text-to-image         |
| **Qwen-Image-2512-SDNQ-4bit-dynamic**    | Text-to-Image    | Detailed and coherent image generation    |
| **FLUX.2-klein-4B-SDNQ-4bit-dynamic**      | Image Editing    | Fast, instruction-based image editing     |
| **FLUX.2-klein-9B-SDNQ-4bit-dynamic-svd-r32**| Image Editing    | Higher quality, instruction-based editing |
| **Qwen-Image-Edit-2511-SDNQ-uint4-svd-r32**| Image Editing    | Powerful, instruction-based editing       |

## 🚀 Getting Started

### Prerequisites

- A Windows operating system.
- **Python** installed. You can download it from [python.org](https://www.python.org/downloads/). **Important:** During installation, make sure to check the box that says "Add Python to PATH".
- **NVIDIA GPU (Recommended):** For acceptable performance, an NVIDIA GPU with up-to-date drivers is strongly recommended. The application will fall back to CPU, but it will be extremely slow.
- **Git** to clone the repository (or you can download the code as a ZIP file).

### Installation & Launch

1.  **Clone the Repository:**
    Open a terminal or Command Prompt and run:
    ```bash
    git clone [URL_OF_YOUR_GITHUB_REPO]
    cd [NAME_OF_YOUR_REPO_FOLDER]
    ```

2.  **Run the Launcher:**
    Simply double-click the **`run.bat`** file.

    The first time you run it, this script will:
    - Create an isolated Python virtual environment in a `venv` folder.
    - Perform a one-time installation of all required libraries, including the correct GPU-enabled version of PyTorch. This step will take several minutes and download a few gigabytes of data.

    On every subsequent run, the script will detect the existing setup and launch the application instantly.

3.  **Use the Application:**
    Once the server is running, the script will display a local URL (usually `http://127.0.0.1:7860`). Open this URL in your web browser to start using Studio X!
