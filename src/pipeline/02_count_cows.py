import cv2
import torch
import supervision as sv
from transformers import DetrImageProcessor, DetrForObjectDetection
from PIL import Image
import numpy as np
import os
from tqdm import tqdm

# --- CONFIGURATION ---
SOURCE_VIDEO_PATH = "data/videos/cows.mp4"
TARGET_VIDEO_PATH = "data/output/cows_counted_debug.mp4"
CONFIDENCE_THRESHOLD = 0.4 

# 🛠️ DEBUGGING FLAG
# Set to True first to rule out GPU driver hangs on Mac
FORCE_CPU = True

ALLOWED_LABELS = ['bird', 'sheep', 'cow', 'bear', 'dog', 'horse', 'zebra']
MIN_AREA_THRESHOLD = 4000 
MAX_AREA_THRESHOLD = 150000 

def run_cow_counting_manual():
    print("🐮 Initializing Manual Cow Counter (Debug Mode)...")

    # 1. Device Setup
    if FORCE_CPU:
        device = torch.device("cpu")
        print("   [!] FORCING CPU MODE for debugging stability.")
    else:
        device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        print(f"   [+] Device: {device}")

    # 2. Load Model
    print("   [+] Loading DETR Model...")
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
    model.to(device)
    model.eval()

    # 3. Logic Setup
    video_info = sv.VideoInfo.from_video_path(SOURCE_VIDEO_PATH)
    w, h = video_info.width, video_info.height
    print(f"   [+] Video Resolution: {w}x{h}")

    tracker = sv.ByteTrack(
        track_activation_threshold=0.25, 
        lost_track_buffer=30, 
        minimum_matching_threshold=0.8, 
        frame_rate=video_info.fps
    )

    line_start = sv.Point(0, h // 2)
    line_end = sv.Point(w, h // 2)
    line_zone = sv.LineZone(start=line_start, end=line_end)

    box_annotator = sv.BoxAnnotator(thickness=2, color=sv.ColorPalette.DEFAULT)
    label_annotator = sv.LabelAnnotator(text_scale=0.5, text_padding=5)
    trace_annotator = sv.TraceAnnotator(thickness=2, trace_length=50)
    line_zone_annotator = sv.LineZoneAnnotator(thickness=2, text_thickness=2, text_scale=1)

    # 4. MANUAL PROCESSING LOOP
    # We open the video manually with supervision's VideoSink (handles codecs well)
    # but we control the loop iteration.
    
    print("🚀 Starting Main Loop (Frame by Frame)...")
    
    # Generator to read frames
    frame_generator = sv.get_video_frames_generator(SOURCE_VIDEO_PATH)
    
    with sv.VideoSink(TARGET_VIDEO_PATH, video_info=video_info) as sink:
        # TQDM creates a progress bar in the terminal
        for i, frame in enumerate(tqdm(frame_generator, total=video_info.total_frames)):
            
            # --- A. INFERENCE ---
            image_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            inputs = processor(images=image_pil, return_tensors="pt").to(device)
            
            with torch.no_grad():
                outputs = model(**inputs)

            target_sizes = torch.tensor([image_pil.size[::-1]]).to(device)
            results = processor.post_process_object_detection(
                outputs, target_sizes=target_sizes, threshold=CONFIDENCE_THRESHOLD
            )[0]

            detections = sv.Detections.from_transformers(transformers_results=results)

            # --- B. FILTERS ---
            valid_indices = []
            for idx, class_id in enumerate(detections.class_id):
                class_name = model.config.id2label[class_id]
                if class_name in ALLOWED_LABELS:
                    valid_indices.append(idx)
            
            if len(valid_indices) > 0:
                detections = detections[np.array(valid_indices)]
            else:
                detections = sv.Detections.empty()

            detections = detections[(detections.area > MIN_AREA_THRESHOLD) & (detections.area < MAX_AREA_THRESHOLD)]

            # --- C. TRACKING ---
            detections = tracker.update_with_detections(detections)

            # --- D. COUNTING ---
            line_zone.trigger(detections=detections)

            # --- E. ANNOTATION ---
            labels = [f"Cow #{tracker_id}" for tracker_id in detections.tracker_id]

            annotated_frame = trace_annotator.annotate(scene=frame.copy(), detections=detections)
            annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections)
            annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
            annotated_frame = line_zone_annotator.annotate(annotated_frame, line_counter=line_zone)

            # Write frame to video file
            sink.write_frame(annotated_frame)

    print(f"\n✅ SUCCESS! Output saved to: {TARGET_VIDEO_PATH}")

if __name__ == "__main__":
    if not os.path.exists(SOURCE_VIDEO_PATH):
        print(f"❌ Error: Video not found at {SOURCE_VIDEO_PATH}")
    else:
        run_cow_counting_manual()