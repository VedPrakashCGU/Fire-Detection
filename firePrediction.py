from ultralytics import YOLO
import cv2

# Load YOLO model (replace 'best.pt' with your trained model if needed)
model = YOLO(r'C:\Users\user\Desktop\FireDetection\runs\detect\train2\weights\best.pt')  # Replace with your model path if custom

# Set detection threshold
threshold = 0.5

# Open the webcam (0 is the default camera, change to 1, 2, etc., for other cameras)
cap = cv2.VideoCapture(1)

# Check if the webcam is opened successfully
if not cap.isOpened():
    print("Error: Unable to access the camera.")
    exit()

# Process live camera feed
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Unable to read the camera feed.")
        break

    # Run YOLO detection on the frame
    results = model(frame)[0]

    # Process detection results
    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result
        if score > threshold:  # Apply detection threshold
            # Draw bounding box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            # Add label
            label = f"{results.names[int(class_id)].upper()} ({score:.2f})"
            cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('YOLO Live Detection', frame)

    # Press 'q' to exit the live feed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
