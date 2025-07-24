import cv2
from yolo import YOLODetector

# Initialize YOLO detector
detector = YOLODetector()

# Start video capture (0 = default webcam)
cap = cv2.VideoCapture(0)

# Set width and height if needed
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Optionally resize frame if needed
    # frame = imutils.resize(frame, width=800)

    # Detect people
    people_boxes = detector.detect_people(frame)

    # Draw bounding boxes
    for (x, y, w, h) in people_boxes:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the count
    text = f"People Count: {len(people_boxes)}"
    cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Show output
    cv2.imshow("Real-Time People Counting", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
