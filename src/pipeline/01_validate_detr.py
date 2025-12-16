import cv2
import torch
import supervision as sv
from transformers import DetrImageProcessor, DetrForObjectDetection
from PIL import Image
import numpy as np
import os

# --- CONFIGURATION ---
SOURCE_VIDEO_PATH = "data/videos/cows.mp4"
TARGET_VIDEO_PATH = "data/output/cows_detected.mp4"
CONFIDENCE_THRESHOLD = 0.5

def run_detr_validation():
    print("🐮 Initializing DETR Validation (Fixed)...")

    # 1. Device Setup
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"   [+] Computing Device: {device}")

    # 2. Load Transformer Model
    print("   [+] Loading DETR ResNet-50...")
    try:
        processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
        model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
        model.to(device)
        model.eval()
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    # 3. Annotators
    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)

    # 4. Video Info
    video_info = sv.VideoInfo.from_video_path(SOURCE_VIDEO_PATH)
    print(f"   [+] Video Info: {video_info.width}x{video_info.height} @ {video_info.fps} FPS")

    # 5. Callback Function
    def callback(frame: np.ndarray, index: int) -> np.ndarray:
        # Debug visual para saber que avanza
        if index % 30 == 0:
            print(f"   ... Processing frame {index}")

        # A. Preprocess
        image_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # B. Inference
        inputs = processor(images=image_pil, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)

        # C. Post-process
        target_sizes = torch.tensor([image_pil.size[::-1]]).to(device)
        results = processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=CONFIDENCE_THRESHOLD
        )[0]

        # D. Convert to Supervision Detections (FIXED)
        # En versiones nuevas, 'id_mapping' ya no se pasa aquí.
        # Pasamos solo los resultados crudos.
        detections = sv.Detections.from_transformers(
            transformers_results=results
        )

        # E. Annotate
        # Hacemos el mapeo de ID a Nombre MANUALMENTE aquí
        labels = []
        for class_id, confidence in zip(detections.class_id, detections.confidence):
            # Obtener nombre desde la config del modelo
            class_name = model.config.id2label[class_id]
            labels.append(f"{class_name} {confidence:.2f}")

        annotated_frame = box_annotator.annotate(
            scene=frame.copy(), detections=detections
        )
        annotated_frame = label_annotator.annotate(
            scene=annotated_frame, detections=detections, labels=labels
        )

        return annotated_frame

    # 6. Run
    print("🚀 Starting Batch Inference...")
    sv.process_video(
        source_path=SOURCE_VIDEO_PATH,
        target_path=TARGET_VIDEO_PATH,
        callback=callback
    )

    print(f"\n✅ Processing complete. Check output at: {TARGET_VIDEO_PATH}")

if __name__ == "__main__":
    if not os.path.exists("data/output"): os.makedirs("data/output")
    if not os.path.exists(SOURCE_VIDEO_PATH):
        print(f"❌ Error: Video not found at {SOURCE_VIDEO_PATH}")
    else:
        run_detr_validation()