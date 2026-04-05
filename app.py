import cv2
from ultralytics import YOLO
import numpy as np
import gradio as gr
import os

# Load model once at startup
print("Loading YOLO model...")
model = YOLO('yolov8n.pt')
print("Model loaded!")

def detect_blue_car(car_crop):
    """Helper function to detect if a car is blue"""
    if car_crop.size == 0:
        return False, 0
    
    hsv = cv2.cvtColor(car_crop, cv2.COLOR_BGR2HSV)
    
    # Mask 1 - Bright/normal blue
    lower_blue1 = np.array([100, 80, 50])
    upper_blue1 = np.array([125, 255, 255])
    mask1 = cv2.inRange(hsv, lower_blue1, upper_blue1)
    
    # Mask 2 - Dark blue only
    lower_blue2 = np.array([100, 60, 20])
    upper_blue2 = np.array([130, 255, 80])
    mask2 = cv2.inRange(hsv, lower_blue2, upper_blue2)
    
    # Combine both masks
    mask = cv2.bitwise_or(mask1, mask2)
    h, w = mask.shape
    center_mask = mask[h//4:3*h//4, w//4:3*w//4]
    blue_ratio = np.sum(center_mask > 0) / center_mask.size
    
    is_blue = blue_ratio > 0.30
    return is_blue, blue_ratio

def process_image(image_path):
    """Process a single image"""
    frame = cv2.imread(image_path)
    
    if frame is None:
        return None, "ERROR: Could not load image"
    
    results = model(frame, conf=0.35, verbose=False)
    result = results[0]
    
    blue_cars = 0
    other_cars = 0
    people = 0
    
    for box in result.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        
        if cls == 2:  # Car
            car_crop = frame[y1:y2, x1:x2]
            is_blue, blue_ratio = detect_blue_car(car_crop)
            
            if is_blue:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.putText(frame, f'Blue Car {conf:.0%}', (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                blue_cars += 1
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 3)
                cv2.putText(frame, f'Car {conf:.0%}', (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                other_cars += 1
        
        elif cls == 0:  # Person
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'Person {conf:.0%}', (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            people += 1
        
    
    # Count display
    cv2.rectangle(frame, (0, 0), (280, 100), (0, 0, 0), -1)
    cv2.putText(frame, f'Cars: {other_cars}', (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f'Blue Cars: {blue_cars}', (10, 65),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.putText(frame, f'People: {people}', (10, 100),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    
    # Save output
    output_path = 'output_image.jpg'
    cv2.imwrite(output_path, frame)
    
    # Convert BGR to RGB for Gradio display
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    summary = f"""
    ✅ Detection Complete!
    
    📊 Results:
    other cars: {other_cars}
    Blue Cars: {blue_cars}
    Total Cars: {other_cars + blue_cars}
    People: {people}
    """
    
    return frame_rgb, summary

def process_video(video_path, progress=gr.Progress()):
    """Process a video file"""
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        return None, "ERROR: Could not open video file"
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_path = 'output_video.mp4'
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_number = 0
    
    # Track unique vehicles across frames using center positions
    tracked_cars = set()
    tracked_blue_cars = set()
    tracked_people = set()
    
    # Cumulative totals
    total_cars = 0
    total_blue_cars = 0
    total_people = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_number += 1
        progress((frame_number / total_frames), desc=f"Processing frame {frame_number}/{total_frames}")
        
        results = model(frame, conf=0.5, verbose=False)
        result = results[0]
        
        cars = []
        people = []
        
        for box in result.boxes:
            cls = int(box.cls[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            if cls == 2:
                cars.append((x1, y1, x2, y2))
            elif cls == 0:
                people.append((x1, y1, x2, y2))
        
        frame_blue_cars = 0
        frame_other_cars = 0
        
        for i, (x1, y1, x2, y2) in enumerate(cars):
            # Calculate center point for tracking
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            
            car_crop = frame[y1:y2, x1:x2]
            is_blue, blue_ratio = detect_blue_car(car_crop)
            
            # Track unique vehicles (with tolerance of 50 pixels for same vehicle)
            car_id = (center_x // 50, center_y // 50)
            
            if is_blue:
                color_box = (0, 0, 255)
                label = 'Blue Car'
                frame_blue_cars += 1
                if car_id not in tracked_blue_cars:
                    tracked_blue_cars.add(car_id)
                    total_blue_cars += 1
            else:
                color_box = (255, 0, 0)
                label = 'Car'
                frame_other_cars += 1
                if car_id not in tracked_cars:
                    tracked_cars.add(car_id)
                    total_cars += 1
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color_box, 3)
            cv2.putText(frame, label, (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_box, 2)
        
        for x1, y1, x2, y2 in people:
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            person_id = (center_x // 50, center_y // 50)
            
            if person_id not in tracked_people:
                tracked_people.add(person_id)
                total_people += 1
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, 'Person', (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Display TOTAL counts (accumulated)
        cv2.rectangle(frame, (0, 0), (320, 120), (0, 0, 0), -1)
        cv2.putText(frame, f'Total Cars: {total_cars}', (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f'Total Blue Cars: {total_blue_cars}', (10, 65),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(frame, f'Total People: {total_people}', (10, 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f'Frame: {frame_number}/{total_frames}', (10, 135),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
        out.write(frame)
    
    cap.release()
    out.release()
    
    summary = f"""
    ✅ Video Processing Complete!
    
    📊 Final Results:
    • Total Cars Detected: {total_cars}
    • Total Blue Cars Detected: {total_blue_cars}
    • Total People Detected: {total_people}
    • Total Frames Processed: {frame_number}
    • Video Dimensions: {width}x{height} @ {fps} fps
    • Output saved to: {output_path}
    """
    
    return output_path, summary

def detect_Car(file):
    """Main function that automatically detects file type and processes accordingly"""
    if file is None:
        return None, "Please upload a file first!"
    
    file_path = file.name
    file_ext = os.path.splitext(file_path)[1].lower()
    
    # Image extensions
    image_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
    # Video extensions
    video_exts = ['.mp4', '.avi', '.mov', '.mkv', '.flv']
    
    if file_ext in image_exts:
        return process_image(file_path)
    elif file_ext in video_exts:
        return process_video(file_path)
    else:
        return None, f"Unsupported file format: {file_ext}"

# Create Gradio Interface
with gr.Blocks(title="Car Detection System") as demo:
    gr.Markdown("""
    # 🚗 Car Detection System
    ### Detects cars, blue cars, and  people in images and videos
    
    **Supported formats:**
    - Images: .jpg, .jpeg, .png, .bmp, .webp
    - Videos: .mp4, .avi, .mov, .mkv, .flv
    """)
    
    with gr.Row():
        with gr.Column():
            input_file = gr.File(label="Upload Image or Video", file_types=["image", "video"])
            detect_btn = gr.Button("🔍 Detect", variant="primary", size="lg")
        
        with gr.Column():
            output_display = gr.Image(label="Detected Output (Image)", type="numpy")
            output_video = gr.Video(label="Detected Output (Video)", visible=False)
            output_text = gr.Textbox(label="Detection Results", lines=8)
    
    def process_and_display(file):
        result, summary = detect_Car(file)
        
        if result is None:
            return None, None, summary
        
        # Check if result is video path (string) or image (numpy array)
        if isinstance(result, str):
            return gr.Image(visible=False), gr.Video(value=result, visible=True), summary
        else:
            return gr.Image(value=result, visible=True), gr.Video(visible=False), summary
    
    detect_btn.click(
        fn=process_and_display,
        inputs=[input_file],
        outputs=[output_display, output_video, output_text]
    )
    
    gr.Markdown("""
    ### 📝 Instructions:
    1. Click "Upload Image or Video" and select your file
    2. Click "🔍 Detect" button
    3. Wait for processing (videos may take a few minutes)
    4. View results with bounding boxes and counts
    
    **Color coding:**
    - 🔴 Red boxes = Blue cars
    - 🔵 Blue boxes = Other cars
    - 🟢 Green boxes = People
    """)

# Launch the app
if __name__ == "__main__":
    demo.launch()  # Public link enabled!
