import cv2
import mediapipe as mp     # 載入 mediapipe 函式庫

cap = cv2.VideoCapture(2)
mp_face_detection = mp.solutions.face_detection   # 建立偵測方法
mp_drawing = mp.solutions.drawing_utils           # 建立繪圖方法

# 開始使用人臉偵測模型
with mp_face_detection.FaceDetection(             # 開始偵測人臉
    model_selection=0, min_detection_confidence=0.5) as face_detection:

    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    # 持續從攝像頭讀取影像
    while True:
        ret, img = cap.read()
        if not ret:
            print("Cannot receive frame")
            break
        # type:numpy.ndarray
        img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   #將BGR顏色轉換成RGB，因為mediapipe 接受的是 RGB 格式的影像
        
        results = face_detection.process(img2)        ##使用人臉偵測模型處理影像，進行人臉偵測
     
        if results.detections:
            for detection in results.detections:
                mp_drawing.draw_detection(img, detection)  #在影像上標記出偵測到的人臉
                # 取出左右眼的標記點位置
                left_eye_landmark = mp_face_detection.get_key_point(
                    detection, mp_face_detection.FaceKeyPoint.LEFT_EYE)
                right_eye_landmark = mp_face_detection.get_key_point(
                    detection, mp_face_detection.FaceKeyPoint.RIGHT_EYE)
                
                # 計算左眼和右眼中心點的座標
                if left_eye_landmark and right_eye_landmark:
                    left_eye_x = int(left_eye_landmark.x * img.shape[1])
                    left_eye_y = int(left_eye_landmark.y * img.shape[0])
                    right_eye_x = int(right_eye_landmark.x * img.shape[1])
                    right_eye_y = int(right_eye_landmark.y * img.shape[0])
                    eye_center_x = (left_eye_x + right_eye_x) // 2
                    eye_center_y = (left_eye_y + right_eye_y) // 2
                    print("Eye center coordinates (x, y):", eye_center_x, eye_center_y)

        cv2.imshow('oxxostudio', img)
        if cv2.waitKey(5) == ord('q'):
            break    # 按下 q 鍵停止
cap.release()
cv2.destroyAllWindows()
