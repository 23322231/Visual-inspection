import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# 啟動攝影機
cap = cv2.VideoCapture(0)

def calculate_direction(finger_tip, finger_root):
    #計算手指向量
    direction_vector = np.array(finger_tip) - np.array(finger_root)

    #定義上下左右向量
    reference_vectors = {
        "up": np.array([0, -1]),
        "down": np.array([0, 1]),
        "right": np.array([-1, 0]),
        "left": np.array([1, 0])
    }

    min_angle = 180  #設定初始角度
    detected_direction = None

    #計算與四個方向向量的夾角
    for direction, ref_vector in reference_vectors.items():
        #cos角度
        cos_angle = np.dot(direction_vector[:2], ref_vector) / (
            np.linalg.norm(direction_vector[:2]) * np.linalg.norm(ref_vector)
        )
        angle= np.degrees(np.arccos(cos_angle))  #轉換為角度

        if angle < min_angle:  #找到最小夾角
            min_angle = angle
            detected_direction = direction

    return detected_direction

#偵測手掌
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:

    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    while True:
        ret, img = cap.read()
        if not ret:
            print("Cannot receive frame")
            break

        #BGR轉RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                

                #取得食指(8)和(5)的座標
                finger_tip = [
                    hand_landmarks.landmark[8].x * img.shape[1],
                    hand_landmarks.landmark[8].y * img.shape[0]
                ]
                finger_root = [
                    hand_landmarks.landmark[5].x * img.shape[1],
                    hand_landmarks.landmark[5].y * img.shape[0]
                ]

                #計算方向
                direction = calculate_direction(finger_tip, finger_root)
                if direction:
                    cv2.putText(
                        img, f"Direction: {direction}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA
                    )

        
        cv2.imshow('Hand Direction Detection', img)
        if cv2.waitKey(5) == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
