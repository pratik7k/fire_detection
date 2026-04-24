import cv2
from ultralytics import YOLO
from dectetion import FireExpertSystem

# 1. Initialize Model and Expert System
model = YOLO('best.pt') # Your trained weights
expert = FireExpertSystem(fps=30)
cap = cv2.VideoCapture('fire.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    # 2. YOLO Inference
    results = model(frame, stream=True, device=0, verbose=False)
    
    found = False
    best_box = None
    max_conf = 0

    for r in results:
        if len(r.boxes) > 0:
            found = True
            # Get data for the largest/most confident box
            i = r.boxes.conf.argmax()
            max_conf = float(r.boxes.conf[i])
            best_box = r.boxes.xywh[i].cpu().numpy() # [x, y, w, h] normalized

    # 3. Update Expert System
    expert.update(found, best_box, max_conf)
    fire_score = expert.get_fire_score()

    # 4. Visualization
    status = "SCANNING"
    color = (0, 255, 0)
    
    if fire_score >= 70: # Confirmed Fire
        status = "CRITICAL: FIRE DETECTED"
        color = (0, 0, 255)
    elif fire_score > 20: # Warning
        status = "WARNING: UNSTABLE HEAT SOURCE"
        color = (0, 255, 255)

    # UI Overlay
    cv2.rectangle(frame, (10, 10), (450, 80), (0,0,0), -1)
    cv2.putText(frame, f"STATUS: {status}", (20, 40), 1, 1.5, color, 2)
    cv2.putText(frame, f"CONFIDENCE SCORE: {fire_score}", (20, 70), 1, 1.2, (255,255,255), 1)

    cv2.imshow("AIT Fire Detection System", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()