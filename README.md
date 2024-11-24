# PaliGemma: Vision-Language Model Implementation

This repository contains an implementation of PaliGemma, a powerful vision-language model that combines Google's Gemma language model with vision capabilities. The model can process both images and text, generating natural language descriptions and engaging in visual conversations.

## 📝 Paper

This implementation is based on the research paper:
[PaLI-X: On Vision-Language Models and Beyond](https://arxiv.org/abs/2407.07726)

## 🚀 Features

- Multi-modal processing (images + text)
- Based on Google's Gemma architecture
- Supports various inference parameters
- CPU and GPU inference support
- Clean, modular implementation

## 📦 Installation

1. Clone this repository:
```bash
git clone https://github.com/arisylafeta/Paligemma-from-scratch
cd Paligemma-from-scratch
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the model weights from [HuggingFace](https://huggingface.co/google/paligemma-3b-pt-448)

## 🛠️ Usage

1. Configure the model parameters in `launch_inference.sh`:
```bash
MODEL_PATH="path/to/your/model"  # Path to downloaded weights
PROMPT="your prompt here"        # Text prompt
IMAGE_FILE_PATH="path/to/image"  # Input image path
MAX_TOKENS_TO_GENERATE=100       # Maximum generation length
TEMPERATURE=0.8                  # Generation temperature (0-1)
TOP_P=0.9                       # Top-p sampling parameter
DO_SAMPLE="False"               # Whether to use sampling
ONLY_CPU="False"                # Force CPU inference
```

2. Run inference:
```bash
bash launch_inference.sh
```

The model will process the image and complete the prompt with a natural description.

## 🏗️ Project Structure

- `modeling_gemma.py`: Core model architecture
- `inference.py`: Inference pipeline
- `utils.py`: Utility functions
- `launch_inference.sh`: Inference script
- `requirements.txt`: Project dependencies

## 🙏 Acknowledgments

Special thanks to:
- [Umar Jamil](https://www.youtube.com/@umarjamilai) for his excellent [YouTube tutorial](https://www.youtube.com/watch?v=vAmKB7iPkWw) that made this implementation possible and provided valuable insights into vision-language models
