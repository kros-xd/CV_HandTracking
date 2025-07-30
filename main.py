import cv2
import mediapipe as mp
import math
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.6
)
mp_drawing = mp.solutions.drawing_utils

# OpenCV video capture
cap = cv2.VideoCapture(0)

# Create blank canvas
canvas = None
prev_point = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for a mirror view
    frame = cv2.flip(frame, 1)
    if canvas is None:
        canvas = np.zeros_like(frame)

    # Convert BGR to RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame to find hands
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Draw landmarks and connections
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
            )

            # Example: Print the coordinates of the tip of the index finger
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            h, w, _ = frame.shape
            x, y = int(index_tip.x * w), int(index_tip.y * h)
            cv2.circle(frame, (x, y), 10, (0, 0, 255), -1)
            cv2.putText(frame, f"Index Tip: ({x}, {y})", (x + 10, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            
            # Pinch Gesture
            thumb = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

            # Gather x,y and w,h values from thumb and index
            x1, y1 = int(thumb.x * w), int(thumb.y * h)
            x2, y2 = int(index.x * w), int(index.y * h)

            # Calculate distance between thumb and index finger
            distance = math.hypot(x2 - x1, y2- y1)

            if distance < 50:

                if prev_point is not None:
                    cv2.circle(frame, (x2, y2), 10, (255, 255, 255), -1) #turns index circle to white, indicating drawing is active
                    cv2.line(canvas, prev_point, (x2, y2), (255, 255, 255), 5) # changes the line color

                prev_point = (x2, y2)

            else:
                prev_point = None  # Reset when not pinching

    # Merge canvas with frame, the values change the merge ratio.
    frame = cv2.addWeighted(frame, 1, canvas, 1, 0)

    # Show the result
    cv2.imshow("Pinch Drawing", frame)

    # Show the frame
    cv2.imshow("Hand Tracking", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
hands.close()
