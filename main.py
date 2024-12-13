import cv2
import mediapipe as mp
import numpy as np
import os
import csv

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

if not os.path.exists("hand_images"):
    os.makedirs("hand_images")
if not os.path.exists("processed_images"):
    os.makedirs("processed_images")
if not os.path.exists("landmarks"):
    os.makedirs("landmarks")

def classical_image_processing(hand_image, frame_index):
    gray = cv2.cvtColor(hand_image, cv2.COLOR_BGR2GRAY)

    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    edges = cv2.Canny(binary, 100, 200)

    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(binary, kernel, iterations=1)
    eroded = cv2.erode(dilated, kernel, iterations=1)

    cv2.imwrite(f"processed_images/binary_{frame_index}.png", binary)
    cv2.imwrite(f"processed_images/edges_{frame_index}.png", edges)
    cv2.imwrite(f"processed_images/eroded_{frame_index}.png", eroded)

    return binary, edges, eroded

def save_landmarks_to_csv(frame_index, hand_landmarks):
    with open(f"landmarks/frame_{frame_index}.csv", 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['x', 'y', 'z'])
        for lm in hand_landmarks.landmark:
            writer.writerow([lm.x, lm.y, lm.z])

cap = cv2.VideoCapture(0)
frame_index = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Nie udało się odczytać obrazu.")
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            h, w, _ = frame.shape
            x_coords = [int(lm.x * w) for lm in hand_landmarks.landmark]
            y_coords = [int(lm.y * h) for lm in hand_landmarks.landmark]
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)

            hand_image = frame[y_min:y_max, x_min:x_max]
            if hand_image.size > 0:
                cv2.imwrite(f"hand_images/hand_{frame_index}.png", hand_image)

                classical_image_processing(hand_image, frame_index)

            save_landmarks_to_csv(frame_index, hand_landmarks)

    cv2.imshow('Gesture Recognition', frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        print("Zatrzymano przetwarzanie.")
        break

    frame_index += 1

cap.release()
cv2.destroyAllWindows()
