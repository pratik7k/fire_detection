from ultralytics import YOLO
import cv2

# 1. Load the model you downloaded from Colab
model = YOLO('best.pt')

# 2. Open the webcam (0 is usually the default laptop camera)
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if success:
        # Run YOLOv8 inference on the frame
        # device=0 tells it to use your RTX 3050
        results = model(frame, stream=True, device=0)

        # Visualize the results on the frame
        for r in results:
            annotated_frame = r.plot()

        # Display the output
        cv2.imshow("RTX 3050 Fire Detection", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()