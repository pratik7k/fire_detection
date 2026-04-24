from ultralytics import YOLO
import cv2
from score import FireExpertSystem

model = YOLO('best.pt')
expert = FireExpertSystem(fps=30)
cap = cv2.VideoCapture('fire.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    # DECREASED THRESHOLD: conf=0.15 (standard is 0.25)
    results = model.predict(frame, stream=True, device=0, conf=0.25, verbose=False)
    
    found = False
    best_box, max_conf = None, 0

    for r in results:
        if len(r.boxes) > 0:
            found = True
            i = r.boxes.conf.argmax()
            max_conf = float(r.boxes.conf[i])
            best_box = r.boxes.xywh[i].cpu().numpy()

    # Feed data into the Expert System
    expert.update(found, best_box, max_conf)
    
    # Get direct return from the system
    analysis = expert.get_fire_status(alarm_threshold=50) # Very sensitive threshold

    if analysis["fire_detected"]:
        # Direct Action (e.g., trigger an API, sound a buzzer, or log)
        print(f"🔥 ALERT: {analysis['reasons']}")
    
    # Still useful to see for debugging
    cv2.putText(frame, f"Score: {analysis['score']}", (20, 50), 1, 2, (0, 0, 255), 2)
    cv2.imshow("Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()