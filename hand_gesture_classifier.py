# import cv2
# import mediapipe as mp
# import numpy as np
# import os
# import csv
#
# # Inicjalizacja MediaPipe
# mp_hands = mp.solutions.hands
# hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
# mp_drawing = mp.solutions.drawing_utils
#
# # Ustawienia projektu
# MAX_SAMPLES_PER_GESTURE = 15  # Maksymalna liczba zapisów na gest
# gesture_counter = 0  # Licznik gestów
# samples_collected = {}  # Licznik zapisanych próbek dla każdego gestu
# current_gesture = None  # Aktualnie wybrany gest
# BBOX_MARGIN = 20  # Margines dla bounding box
#
# # Tworzenie katalogów
# def create_gesture_directory(gesture_name):
#     os.makedirs(f"hand_images/{gesture_name}", exist_ok=True)
#     os.makedirs(f"processed_images/{gesture_name}/binary", exist_ok=True)
#     os.makedirs(f"processed_images/{gesture_name}/edges", exist_ok=True)
#     os.makedirs(f"processed_images/{gesture_name}/eroded", exist_ok=True)
#     os.makedirs(f"landmarks/{gesture_name}", exist_ok=True)
#
# # Funkcja do klasycznego przetwarzania obrazu
# def classical_image_processing(hand_image, gesture, sample_index):
#     gray = cv2.cvtColor(hand_image, cv2.COLOR_BGR2GRAY)
#     _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#     edges = cv2.Canny(binary, 100, 200)
#     kernel = np.ones((3, 3), np.uint8)
#     dilated = cv2.dilate(binary, kernel, iterations=1)
#     eroded = cv2.erode(dilated, kernel, iterations=1)
#
#     # Zapis przetworzonych obrazów do folderu gestu
#     cv2.imwrite(f"processed_images/{gesture}/binary/binary_{sample_index}.png", binary)
#     cv2.imwrite(f"processed_images/{gesture}/edges/edges_{sample_index}.png", edges)
#     cv2.imwrite(f"processed_images/{gesture}/eroded/eroded_{sample_index}.png", eroded)
#
# # Funkcja zapisu punktów charakterystycznych do pliku CSV
# def save_landmarks_to_csv(gesture, sample_index, hand_landmarks):
#     with open(f"landmarks/{gesture}/frame_{sample_index}.csv", 'w', newline='') as csvfile:
#         writer = csv.writer(csvfile)
#         writer.writerow(['x', 'y', 'z'])  # Nagłówki
#         for lm in hand_landmarks.landmark:
#             writer.writerow([lm.x, lm.y, lm.z])  # Zapis współrzędnych punktów
#
# # Kamera
# cap = cv2.VideoCapture(0)
#
# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         print("Nie udało się odczytać obrazu.")
#         break
#
#     # Konwersja obrazu do RGB (wymagane przez MediaPipe)
#     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     results = hands.process(frame_rgb)
#
#     # Naciśnięcie klawisza dodaje nowy gest
#     key = cv2.waitKey(10) & 0xFF
#     if key != 255:  # Sprawdzamy, czy wciśnięto klawisz
#         gesture_counter += 1
#         current_gesture = f"gest_{gesture_counter}"
#         samples_collected[current_gesture] = 0
#         create_gesture_directory(current_gesture)
#         print(f"Dodano nowy gest: {current_gesture}")
#
#     # Jeśli wykryto dłoń i wybrano gest
#     if results.multi_hand_landmarks and current_gesture:
#         for hand_landmarks in results.multi_hand_landmarks:
#             # Ograniczenie liczby zapisów na gest
#             if samples_collected[current_gesture] >= MAX_SAMPLES_PER_GESTURE:
#                 continue
#
#             # Wyznaczenie obszaru dłoni (bounding box) z marginesem
#             h, w, _ = frame.shape
#             x_coords = [int(lm.x * w) for lm in hand_landmarks.landmark]
#             y_coords = [int(lm.y * h) for lm in hand_landmarks.landmark]
#             x_min, x_max = max(min(x_coords) - BBOX_MARGIN, 0), min(max(x_coords) + BBOX_MARGIN, w)
#             y_min, y_max = max(min(y_coords) - BBOX_MARGIN, 0), min(max(y_coords) + BBOX_MARGIN, h)
#             hand_image = frame[y_min:y_max, x_min:x_max]
#
#             # Zapis danych
#             if hand_image.size > 0:
#                 sample_index = samples_collected[current_gesture]
#
#                 # Zapis obrazu dłoni (bez landmarks)
#                 cv2.imwrite(f"hand_images/{current_gesture}/hand_{sample_index}.png", hand_image)
#
#                 # Przetwarzanie obrazu i zapis
#                 classical_image_processing(hand_image, current_gesture, sample_index)
#
#                 # Zapis punktów charakterystycznych
#                 save_landmarks_to_csv(current_gesture, sample_index, hand_landmarks)
#
#                 # Aktualizacja liczby zapisanych próbek
#                 samples_collected[current_gesture] += 1
#
#             # Rysowanie landmarks na obrazie do wyświetlania (ale nie zapisu)
#             mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
#
#     # Wyświetlanie obrazu w oknie
#     cv2.imshow('Gesture Recognition', frame)
#
#     # Wyjście z programu po naciśnięciu 'q'
#     if key == ord('q'):
#         print("Zatrzymano przetwarzanie.")
#         break
#
# cap.release()
# cv2.destroyAllWindows()


import cv2
import mediapipe as mp
import numpy as np
import os
import csv

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

MAX_SAMPLES_PER_GESTURE = 15
gesture_counter = 0
samples_collected = {}
current_gesture = None
BBOX_MARGIN = 20


def create_gesture_directory(gesture_name):
    os.makedirs(f"hand_images/{gesture_name}", exist_ok=True)
    os.makedirs(f"processed_images/{gesture_name}/binary", exist_ok=True)
    os.makedirs(f"processed_images/{gesture_name}/edges", exist_ok=True)
    os.makedirs(f"processed_images/{gesture_name}/eroded", exist_ok=True)
    os.makedirs(f"landmarks/{gesture_name}", exist_ok=True)


def calculate_distance(point1, point2):
    return np.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2)

def classify_gesture(hand_landmarks):

    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

    index_pip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP]
    middle_pip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
    ring_pip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP]
    pinky_pip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP]

    thumb_ip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP]
    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]

    def is_finger_bent(tip, pip):
        return tip.y > pip.y

    def is_finger_straight(tip, pip):
        return tip.y < pip.y

    # 1. OK Gesture: Kciuk i palec wskazujący tworzą okrąg
    thumb_to_index = calculate_distance(thumb_tip, index_tip)
    if thumb_to_index < 0.05 and middle_tip.y > index_tip.y:
        return "OK"

    # 2. Call Gesture: Kciuk i mały palec wyprostowane, reszta zgięta
    if is_finger_straight(thumb_tip, thumb_ip) and is_finger_straight(pinky_tip, pinky_pip) and \
            all(is_finger_bent(tip, pip) for tip, pip in [(index_tip, index_pip), (middle_tip, middle_pip), (ring_tip, ring_pip)]):
        return "Call"

    # 3. Peace Gesture: Wskazujący i środkowy wyprostowane, reszta zgięta
    if is_finger_straight(index_tip, index_pip) and is_finger_straight(middle_tip, middle_pip) and \
            all(is_finger_bent(tip, pip) for tip, pip in [(ring_tip, ring_pip), (pinky_tip, pinky_pip)]):
        return "Peace"

    # 4. Rock Gesture: Kciuk, wskazujący i mały palec wyprostowane
    if is_finger_straight(thumb_tip, thumb_ip) and is_finger_straight(index_tip, index_pip) and \
            is_finger_straight(pinky_tip, pinky_pip) and \
            all(is_finger_bent(tip, pip) for tip, pip in [(middle_tip, middle_pip), (ring_tip, ring_pip)]):
        return "Rock"

    # 5. Thumbs Up Gesture: Kciuk wyprostowany, reszta zgięta
    if is_finger_straight(thumb_tip, thumb_ip) and \
            all(is_finger_bent(tip, pip) for tip, pip in [(index_tip, index_pip), (middle_tip, middle_pip), (ring_tip, ring_pip), (pinky_tip, pinky_pip)]):
        return "Thumbs Up"

    # 6. Fist Gesture: Wszystkie palce zgięte
    if all(is_finger_bent(tip, pip) for tip, pip in [(index_tip, index_pip), (middle_tip, middle_pip), (ring_tip, ring_pip), (pinky_tip, pinky_pip)]):
        return "Fist"

    # 7. Open Palm Gesture: Wszystkie palce wyprostowane
    if all(is_finger_straight(tip, pip) for tip, pip in [(index_tip, index_pip), (middle_tip, middle_pip), (ring_tip, ring_pip), (pinky_tip, pinky_pip)]):
        return "Palm"

    # Domyślnie: Nieznany gest
    return "Unknown Gesture"



def classical_image_processing(hand_image, gesture, sample_index):
    gray = cv2.cvtColor(hand_image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    edges = cv2.Canny(binary, 100, 200)
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(binary, kernel, iterations=1)
    eroded = cv2.erode(dilated, kernel, iterations=1)

    cv2.imwrite(f"processed_images/{gesture}/binary/binary_{sample_index}.png", binary)
    cv2.imwrite(f"processed_images/{gesture}/edges/edges_{sample_index}.png", edges)
    cv2.imwrite(f"processed_images/{gesture}/eroded/eroded_{sample_index}.png", eroded)


def save_landmarks_to_csv(gesture, sample_index, hand_landmarks):
    with open(f"landmarks/{gesture}/frame_{sample_index}.csv", 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['x', 'y', 'z'])
        for lm in hand_landmarks.landmark:
            writer.writerow([lm.x, lm.y, lm.z])


cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Nie udało się odczytać obrazu.")
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    key = cv2.waitKey(10) & 0xFF
    if key != 255:
        gesture_counter += 1
        current_gesture = f"gest_{gesture_counter}"
        samples_collected[current_gesture] = 0
        create_gesture_directory(current_gesture)
        print(f"Dodano nowy gest: {current_gesture}")

    if results.multi_hand_landmarks and current_gesture:
        for hand_landmarks in results.multi_hand_landmarks:
            if samples_collected[current_gesture] >= MAX_SAMPLES_PER_GESTURE:
                continue

            gesture_name = classify_gesture(hand_landmarks)
            cv2.putText(frame, gesture_name, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            h, w, _ = frame.shape
            x_coords = [int(lm.x * w) for lm in hand_landmarks.landmark]
            y_coords = [int(lm.y * h) for lm in hand_landmarks.landmark]
            x_min, x_max = max(min(x_coords) - BBOX_MARGIN, 0), min(max(x_coords) + BBOX_MARGIN, w)
            y_min, y_max = max(min(y_coords) - BBOX_MARGIN, 0), min(max(y_coords) + BBOX_MARGIN, h)
            hand_image = frame[y_min:y_max, x_min:x_max]

            if hand_image.size > 0:
                sample_index = samples_collected[current_gesture]

                cv2.imwrite(f"hand_images/{current_gesture}/hand_{sample_index}.png", hand_image)

                classical_image_processing(hand_image, current_gesture, sample_index)

                save_landmarks_to_csv(current_gesture, sample_index, hand_landmarks)

                samples_collected[current_gesture] += 1

            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow('Gesture Recognition', frame)

    if key == ord('q'):
        print("Zatrzymano przetwarzanie.")
        break

cap.release()
cv2.destroyAllWindows()
