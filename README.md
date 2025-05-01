# XAI-VNPlant200: Explainable AI for Plant Species Identification

This project leverages deep learning models and Explainable AI (XAI) techniques to identify plant species and provide interpretability for the predictions. It includes a web application for interactive demonstrations and Jupyter notebooks for experimentation.

## Features

- **Deep Learning**: InceptionV3 and Vision Transformer (ViT) for plant species classification.
- **Explainability**: LIME and Grad-CAM techniques for model interpretability.
- **Segmentation**: Quickshift-based image segmentation for preprocessing and visualization.

---

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- (Optional) CUDA-enabled GPU for faster inference

### Steps

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-repo/xai-vnplant200.git
   cd xai-vnplant200

2. **Set Up a Virtual Environment**:

python -m venv env
source env/bin/activate

3. **Install Dependencies**:

pip install -r requirements.txt

4. **Refer to xai-inceptionv3.ipynb to fine-tune InceptionV3**