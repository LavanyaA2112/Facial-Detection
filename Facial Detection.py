import cv2
import os
import datetime

# Create a folder to save detected face images
output_folder = "detected_faces"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Load Haar cascades for face and eye detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

# Choose source: 0 for webcam, or path to image
use_webcam = True
image_path = "your_image.jpg"  # Use only if use_webcam = False

if use_webcam:
    cap = cv2.VideoCapture(0)
else:
    frame = cv2.imread(image_path)
    cap = None

frame_count = 0

while True:
    # Read from webcam or image
    if use_webcam:
        ret, frame = cap.read()
        if not ret:
            break
    else:
        if frame_count > 0:
            break  # show image only once if using image file
        frame_count += 1

    # Convert to grayscale for detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # Loop through faces and draw rectangles
    for i, (x, y, w, h) in enumerate(faces):
        face_roi = frame[y:y + h, x:x + w]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Save detected face image
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        face_filename = os.path.join(output_folder, f"face_{timestamp}_{i}.jpg")
        cv2.imwrite(face_filename, face_roi)

        # Detect eyes within the face region
        face_gray = gray[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(face_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(face_roi, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 1)

    # Display the result
    cv2.imshow("Face Detection", frame)

    # Press 'q' to quit
    key = cv2.waitKey(1 if use_webcam else 0)
    if key == ord('q'):
        break

# Release resources
if cap:
    cap.release()
cv2.destroyAllWindows()
