import cv2
from ultralytics import YOLO

model = YOLO('best.pt')
cap = cv2.VideoCapture('fire_video.mp4')

# Get video properties for saving
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define the codec and create VideoWriter object
out = cv2.VideoWriter('output_processed.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Run YOLOv8 inference (stream=True is more memory efficient)
    results = model(frame, stream=True, device=0)

    for r in results:
        annotated_frame = r.plot()  # Draw the boxes
        
    # Write the frame to the output file
    out.write(annotated_frame)

    # Display the frame (Press 'q' to stop)
    cv2.imshow("RTX 3050 Inference", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()