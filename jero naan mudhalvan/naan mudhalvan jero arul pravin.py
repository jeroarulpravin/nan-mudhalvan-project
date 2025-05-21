# Real Time fire and smoke detection - video input

import cv2
import numpy as np

# Initialize video capture from webcam
cap = cv2.VideoCapture(0)

# Read the first frame
ret, prev_frame = cap.read()
prev_frame = cv2.GaussianBlur(prev_frame, (21, 21), 0)
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the current frame
    blurred = cv2.GaussianBlur(frame, (21, 21), 0)
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

    # Frame differencing
    diff = cv2.absdiff(prev_gray, gray)
    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

    # Morphology to remove noise
    thresh = cv2.dilate(thresh, None, iterations=2)

    # Contour detection
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 5000:  # Fire tends to have chaotic large-area movement
            (x, y, w, h) = cv2.boundingRect(contour)
            roi = frame[y:y+h, x:x+w]
            roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(roi_hsv, (0, 50, 50), (35, 255, 255))  # basic fire color mask

            fire_pixels = cv2.countNonZero(mask)
            if fire_pixels / (w * h) > 0.4:  # enough fire-colored pixels in motion area
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(frame, "FIRE!", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    cv2.imshow("Fire Detection", frame)
    prev_gray = gray.copy()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


# Real Time fire and smoke detection - image inputs fire,smoke,firesmoke
import cv2
import numpy as np

def detect_fire_and_smoke(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print("Image not found or path is incorrect.")
        return

    output = image.copy()
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    ### 1. Fire Detection ###
    # HSV Range for Fire (tuned)
    fire_lower = np.array([0, 50, 200])
    fire_upper = np.array([35, 255, 255])
    fire_mask = cv2.inRange(hsv, fire_lower, fire_upper)

    # Morphological filtering
    fire_mask = cv2.morphologyEx(fire_mask, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))

    contours, _ = cv2.findContours(fire_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 1000:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(output, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.putText(output, "FIRE", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)

    ### 2. Smoke Detection ###
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (21, 21), 0)

    # Use Laplacian to detect lack of sharpness (smoke = soft texture)
    laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
    contrast = np.var(laplacian)

    if contrast < 50:  # Empirical threshold
        cv2.putText(output, "SMOKE LIKELY DETECTED", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)

    # Show results
    cv2.imshow("Detection", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

detect_fire_and_smoke("fire.png")
