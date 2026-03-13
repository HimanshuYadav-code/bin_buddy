import cv2
import google.generativeai as genai
import os
import json
import time
import threading
import queue


# Configure Gemini API
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
model = genai.GenerativeModel("gemini-2.5-flash")

# Shared variables for communication between threads
frame_queue = queue.Queue(maxsize=1)
api_text = "Awaiting classification..."
text_lock = threading.Lock()

# Function to handle the API calls in a separate thread
def classify_frame():
    global api_text
    while True:
        try:
            # Get the latest frame from the queue (blocks if empty)
            frame = frame_queue.get()
            
            # Encode frame to JPEG bytes
            _, buffer = cv2.imencode(".jpg", frame)
            img_bytes = buffer.tobytes()

            try:
                # the Gemini API call
                response = model.generate_content([
                    {"mime_type": "image/jpeg", "data": img_bytes},
                    "You are an expert waste classifier. Classify the object into: "
                    "'Recyclable Plastic', 'Recyclable Paper', 'Compost', or 'Landfill'. "
                    "Return valid JSON with keys: object_name, classification, description."
                ])

                text_output = response.text.strip()
                print("Gemini Response:", text_output)

                try:
                    data = json.loads(text_output)
                    obj = data.get("object_name", "Unknown")
                    category = data.get("classification", "N/A")
                    desc = data.get("description", "")
                    
                    # Used a lock to safely update the shared variable
                    with text_lock:
                        api_text = f"Object: {obj} | Category: {category} | Description: {desc}"
                except json.JSONDecodeError:
                    with text_lock:
                        api_text = "⚠️ Could not parse response."

            except Exception as e:
                with text_lock:
                    api_text = f"API Error: {e}"

        except Exception as e:
            print(f"Worker thread error: {e}")
            time.sleep(1) # Wait before retrying

# Start the worker thread
api_thread = threading.Thread(target=classify_frame, daemon=True)
api_thread.start()

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Error: Could not access camera.")
    exit()

# Function to draw multi-line text within a box
def draw_text_in_box(img, text, box_height, padding=10):
    h, w, _ = img.shape
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    font_thickness = 2
    
    # Split text into lines if it's too long
    words = text.split(' ')
    lines = []
    current_line = ''
    for word in words:
        test_line = current_line + ' ' + word if current_line else word
        (text_w, _), _ = cv2.getTextSize(test_line, font, font_scale, font_thickness)
        
        if text_w < w - 2 * padding:
            current_line = test_line
        else:
            lines.append(current_line)
            current_line = word
    lines.append(current_line)

    overlay = img.copy()
    cv2.rectangle(overlay, (0, h - box_height), (w, h), (0, 0, 0), -1)
    img = cv2.addWeighted(overlay, 0.6, img, 0.4, 0)
    
    for i, line in enumerate(lines):
        y = h - box_height + padding + (i * 30)
        cv2.putText(img, line, (padding, y), font, font_scale, (0, 255, 0), font_thickness)
        
    return img

while True:
    # --- Main thread loop ---
    ret, frame = cap.read()
    if not ret:
        break

    # Put the frame in the queue for the worker thread
    # This non-blocking put() ensures the main loop doesn't wait
    if not frame_queue.full():
        frame_queue.put(frame.copy())
        
    # Get the latest text from the worker thread (using a lock for safety)
    with text_lock:
        display_text = api_text
    
    # Draw the text in a flexible box
    frame = draw_text_in_box(frame, display_text, box_height=100)
    
    # Show camera window
    cv2.imshow("🗑️ Waste Classifier Live", frame)

    # Quit on 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
