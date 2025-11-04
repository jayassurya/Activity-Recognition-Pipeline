# demo.py
import torch
import torch.nn.functional as F
import cv2
import numpy as np
from collections import deque, Counter # Import Counter
import os
import matplotlib.pyplot as plt # Import Matplotlib

# --- 1. Import our custom modules ---
from model import create_model
from dataloader import KINETICS_MEAN, KINETICS_STD, NUM_FRAMES, FRAME_HEIGHT, FRAME_WIDTH

# --- 2. Configuration ---
INPUT_VIDEO_PATH = r"UCF101_new\PullUps\v_PullUps_g01_c03.avi" 
OUTPUT_VIDEO_PATH = "demo_output_final.mp4" 
MODEL_PATH = "activity_model_best.pth"
NUM_CLASSES = 7
CONF_THRESHOLD = 0.50 # 50% confidence threshold

CLASS_NAMES = [
    "Archery",
    "Basketball",
    "CricketShot",
    "HorseRiding",
    "PullUps",
    "Surfing",
    "WritingOnBoard",
]

# --- 3. Setup Model and Device ---
print("Loading model and setting up device...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

if len(CLASS_NAMES) != NUM_CLASSES:
    raise ValueError("Error: NUM_CLASSES does not match the length of CLASS_NAMES")

model = create_model(num_classes=NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
model = model.to(device)
print("Model loaded successfully.")

# --- 4. Define Preprocessing ---
def preprocess_frame(frame):
    frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_tensor = torch.from_numpy(frame).float() / 255.0
    frame_tensor = frame_tensor.permute(2, 0, 1) # [H, W, C] -> [C, H, W]
    
    mean = torch.tensor(KINETICS_MEAN, device=device).view(3, 1, 1)
    std = torch.tensor(KINETICS_STD, device=device).view(3, 1, 1)
    frame_tensor = (frame_tensor.to(device) - mean) / std
    
    return frame_tensor

# --- 5. Helper function to draw the bar chart ---
def draw_probabilities(frame, probabilities, class_names, bar_start_y=50, bar_height=20, bar_spacing=5, bar_length_scale=200):
    # (This is your real-time bar chart function)
    h, w, _ = frame.shape
    sorted_probs, sorted_indices = torch.sort(probabilities[0], descending=True)
    
    for i, idx in enumerate(sorted_indices):
        class_name = class_names[idx.item()]
        score = sorted_probs[i].item()
        y_pos = bar_start_y + i * (bar_height + bar_spacing)
        bar_width = int(score * bar_length_scale) 
        
        color = (0, 255, 0) if i == 0 and score > CONF_THRESHOLD else (200, 200, 200) 
        cv2.rectangle(frame, (10, y_pos), (10 + bar_length_scale, y_pos + bar_height), (50, 50, 50), -1)
        cv2.rectangle(frame, (10, y_pos), (10 + bar_width, y_pos + bar_height), color, -1)
        
        text = f"{class_name}: {score*100:.1f}%"
        cv2.putText(frame, text, (15, y_pos + bar_height - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0) if i == 0 and score > CONF_THRESHOLD else (255, 255, 255), 1)

# --- 6. The Inference (Sliding Window) Logic ---
print(f"Starting inference on {INPUT_VIDEO_PATH}...")

frame_buffer = deque(maxlen=NUM_FRAMES)
prediction_buffer = deque(maxlen=15) 

# List for final summary bar chart
all_predictions_list = []

# --- NEW: Variables for Time Segment Logging ---
all_action_segments = []
current_action_for_segment = "Waiting..." # Holds the *current* stable action
current_action_start_time = 0.0

cap = cv2.VideoCapture(INPUT_VIDEO_PATH)
frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (frame_w, frame_h))

current_frame_confidence = 0.0 
all_class_probabilities = torch.zeros(1, NUM_CLASSES).to(device) 
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break 
    
    original_frame = frame.copy() 
    processed_frame = preprocess_frame(frame)
    frame_buffer.append(processed_frame)
    current_time_sec = frame_count / fps # Get current time
    
    final_action_decision = "Waiting..." # Default value
    
    if len(frame_buffer) == NUM_FRAMES:
        
        clip_tensor = torch.stack(list(frame_buffer), dim=0).permute(1, 0, 2, 3).unsqueeze(0)

        with torch.no_grad():
            outputs = model(clip_tensor.to(device))
            all_class_probabilities = F.softmax(outputs, dim=1) 
            
            confidence, predicted_idx = torch.max(all_class_probabilities, 1)
            
            raw_prediction_for_buffer = CLASS_NAMES[predicted_idx.item()]
            current_frame_confidence = confidence.item()

            prediction_buffer.append(raw_prediction_for_buffer)
            
            smoothed_prediction_text = max(set(prediction_buffer), key=prediction_buffer.count)

            # --- Determine final action for this frame (for logging and display) ---
            if current_frame_confidence > CONF_THRESHOLD:
                final_action_decision = smoothed_prediction_text
            else:
                final_action_decision = "Unknown"
        
        # --- NEW: Time Segment Logging Logic ---
        if final_action_decision != current_action_for_segment:
            # Action has changed. Log the previous segment.
            if current_action_for_segment != "Waiting...": # Don't log the initial "Waiting" state
                segment = {
                    'start': current_action_start_time, 
                    'end': current_time_sec, 
                    'action': current_action_for_segment
                }
                all_action_segments.append(segment)
            
            # Start the new segment
            current_action_for_segment = final_action_decision
            current_action_start_time = current_time_sec
        
        # Store prediction for final bar chart
        all_predictions_list.append(final_action_decision)

    # --- Visualization ---
    cv2.rectangle(original_frame, (0, 0), (frame_w, 40), (0, 0, 0), -1)
    display_text = f"Action: {final_action_decision} (Conf: {current_frame_confidence*100:.1f}%)"
    cv2.putText(original_frame, display_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    if len(frame_buffer) == NUM_FRAMES:
        draw_probabilities(original_frame, all_class_probabilities, CLASS_NAMES, bar_start_y=60)
    
    # Print real-time log to terminal
    print(f"Timestamp: {current_time_sec:.2f}s | Action: {final_action_decision}")
    
    out.write(original_frame)
    cv2.imshow('Activity Recognition', original_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
    frame_count += 1

# --- 7. Cleanup ---
print(f"\nInference finished. Output video saved to {OUTPUT_VIDEO_PATH}")
cap.release()
out.release()
cv2.destroyAllWindows()

# --- NEW: Log the very last action segment ---
if current_action_for_segment != "Waiting...":
    segment = {
        'start': current_action_start_time, 
        'end': current_time_sec, # Use the final timestamp
        'action': current_action_for_segment
    }
    all_action_segments.append(segment)


# --- 8. Final Summary Generation ---
print("\n" + "="*50)
print("           VIDEO ACTION SUMMARY")
print("="*50)

# --- NEW: Print the Action Time Segments ---
print("\n--- Action Time Segments ---")
if not all_action_segments:
    print("No complete action segments were detected.")
else:
    for seg in all_action_segments:
        print(f"  {seg['start']:>6.2f}s to {seg['end']:>6.2f}s  =  {seg['action']}")

# --- Print Overall Percentages ---
print("\n--- Overall Prediction Percentages ---")
if not all_predictions_list:
    print("No predictions were made (video might be too short or file issue).")
else:
    total_predictions = len(all_predictions_list)
    prediction_counts = Counter(all_predictions_list)
    
    # Get all unique labels found (including "Unknown")
    all_detected_labels = sorted(prediction_counts.keys())
    class_percentages = []
    
    for label in all_detected_labels:
        count = prediction_counts[label]
        percentage = (count / total_predictions) * 100
        class_percentages.append(percentage)
        print(f"  {label}: {percentage:.2f}%")

    # 3. Find and print the max percentage (your requested output)
    max_percentage = max(class_percentages)
    max_class_index = class_percentages.index(max_percentage)
    max_class_name = all_detected_labels[max_class_index]
    
    print("\n-------------------------------------------")
    print(f"DOMINANT ACTION: {max_class_name} ({max_percentage:.2f}%)")
    print("-------------------------------------------")

    # 4. Generate the bar chart
    plt.figure(figsize=(10, 6)) 
    plt.bar(all_detected_labels, class_percentages, color='skyblue')
    plt.xlabel('Action Class')
    plt.ylabel('Percentage of Video (%)')
    plt.title(f'Action Prediction Summary for {os.path.basename(INPUT_VIDEO_PATH)}')
    plt.xticks(rotation=45, ha='right') 
    plt.tight_layout() 
    
    print("Displaying summary bar chart...")
    plt.show()

print("Script finished.")