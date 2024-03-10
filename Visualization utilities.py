import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils          # mediapipe 繪圖方法
mp_drawing_styles = mp.solutions.drawing_styles  # mediapipe 繪圖樣式
mp_hands = mp.solutions.hands                    # mediapipe 偵測手掌方法

cap = cv2.VideoCapture(2)

# mediapipe 啟用偵測手掌
with mp_hands.Hands(
    model_complexity=0,
    # max_num_hands=1,
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

        # 水平翻轉影像
        img = cv2.flip(img, 1)

        img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   # 將 BGR 轉換成 RGB
        results = hands.process(img2)                 # 偵測手掌
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                

                # 將節點和骨架繪製到影像中
                mp_drawing.draw_landmarks(
                    img,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
                
                
                # 獲取食指指尖和食指根部的坐標
                index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                index_finger_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
                
                tip_x, tip_y = int(index_finger_tip.x * img.shape[1]), int(index_finger_tip.y * img.shape[0])
                mcp_x, mcp_y = int(index_finger_mcp.x * img.shape[1]), int(index_finger_mcp.y * img.shape[0])

                # 判斷食指方向
                if tip_x < mcp_x - 20:
                    direction = "Left"
                elif tip_x > mcp_x + 20:
                    direction = "Right"
                elif tip_y < mcp_y - 20:
                    direction = "Up"
                elif tip_y > mcp_y + 20:
                    direction = "Down"
                else:
                    direction = "None"


                cv2.putText(img, f"Index Finger Direction: {direction}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        
        cv2.imshow('oxxostudio', img)
        if cv2.waitKey(5) == ord('q'):
            break    # 按下 q 鍵停止
cap.release()
cv2.destroyAllWindows()