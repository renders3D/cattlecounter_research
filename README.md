# 🐮 CattleCounter: Batch Video Analysis with Transformers

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![HuggingFace](https://img.shields.io/badge/Transformers-DETR-yellow)
![Supervision](https://img.shields.io/badge/Supervision-Tracking-purple)
![Status](https://img.shields.io/badge/Status-Research_Phase-orange)

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
