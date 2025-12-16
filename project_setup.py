import os

def create_structure():
    structure = {
        "data": ["videos", "output"],
        "models": [],
        "notebooks": [],
        "src": ["pipeline", "utils"]
    }

    project_name = "CattleCounter_Research"
    print(f"🐮 Initializing '{project_name}' environment...")
    
    if not os.path.exists(project_name):
        os.makedirs(project_name)

    for main_folder, subfolders in structure.items():
        path = os.path.join(project_name, main_folder)
        if not os.path.exists(path):
            os.makedirs(path)
            print(f"   [+] Created: {path}/")
        
        for sub in subfolders:
            sub_path = os.path.join(path, sub)
            if not os.path.exists(sub_path):
                os.makedirs(sub_path)
                print(f"   [+] Created: {sub_path}/")

    # Create README
    create_readme(project_name)
    # Create Requirements
    create_requirements(project_name)
    # Create Gitignore
    create_gitignore(project_name)

    print("\n✅ Environment ready. Moo-ving forward!")

def create_readme(base_path):
    content = """# 🐮 CattleCounter: Batch Video Analysis with Transformers

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![HuggingFace](https://img.shields.io/badge/Transformers-DETR-yellow)
![Supervision](https://img.shields.io/badge/Supervision-Tracking-purple)

**CattleCounter** is a Computer Vision research project focused on accurate livestock counting using **Vision Transformers (ViT)**. Unlike real-time CNNs, this project prioritizes accuracy and global context understanding using the **DETR (DEtection TRansformer)** architecture.

## 🎯 Objectives

1.  **Transformer Implementation:** Use Hugging Face's `facebook/detr-resnet-50` to detect cattle.
2.  **Batch Processing:** Process video frames in batches to optimize GPU/CPU throughput.
3.  **Tracking & Counting:** Implement robust tracking algorithms (ByteTrack) via `supervision` to count unique cows crossing a line.

## 🛠️ Tech Stack

* **Model:** DETR (Facebook AI Research)
* **Library:** Hugging Face Transformers
* **Utilities:** Roboflow Supervision (for handling detections)
* **Processing:** PyTorch

---
*Research Lead: Carlos Luis Noriega*
"""
    with open(os.path.join(base_path, "README.md"), "w") as f:
        f.write(content)

def create_requirements(base_path):
    content = """# --- Core AI ---
torch
torchvision
transformers
timm

# --- Video & Image ---
opencv-python
pillow

# --- The CV Swiss Army Knife ---
supervision

# --- Utils ---
tqdm
numpy
"""
    with open(os.path.join(base_path, "requirements.txt"), "w") as f:
        f.write(content)

def create_gitignore(base_path):
    content = """__pycache__/
*.py[cod]
.DS_Store
data/videos/*
data/output/*
venv/
.venv
.env
"""
    with open(os.path.join(base_path, ".gitignore"), "w") as f:
        f.write(content)

if __name__ == "__main__":
    create_structure()