# 測量眼睛距離，目前數據有點奇怪
# 測得視窗中間的距離值
import pyrealsense2 as rs
import numpy as np
import cv2
import mediapipe as mp     # 載入 mediapipe 函式庫

mp_face_detection = mp.solutions.face_detection   # 建立偵測方法
mp_drawing = mp.solutions.drawing_utils           # 建立繪圖方法

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

print(device_product_line)

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# if device_product_line == 'L500':
config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
# else:
    # config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

try:
    # 開始使用人臉偵測模型
    with mp_face_detection.FaceDetection(             # 開始偵測人臉
        model_selection=0, min_detection_confidence=0.5) as face_detection:
        while True:
            eye_center_x=0
            eye_center_y=0
            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames() #等待串流 pipeline 中的一組幀。這些幀通常包括深度影像、彩色影像等
            depth_frame = frames.get_depth_frame()#返回深度影像幀。
            color_frame = frames.get_color_frame()#返回彩色影像幀。
            if not depth_frame or not color_frame:#檢查是否成功獲取了深度影像和彩色影像幀
                continue

            # Convert images to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())#獲取深度影像的資料並轉換為 NumPy 陣列
            color_image = np.asanyarray(color_frame.get_data())#獲取彩色影像的資料並轉換為 NumPy 陣列
            img2=np.asanyarray(color_frame.get_data())
            results = face_detection.process(img2) 

            if results.detections:
                for detection in results.detections:
                    mp_drawing.draw_detection(img2, detection)  #在影像上標記出偵測到的人臉
                    # 取出左右眼的標記點位置
                    left_eye_landmark = mp_face_detection.get_key_point(
                        detection, mp_face_detection.FaceKeyPoint.LEFT_EYE)
                    right_eye_landmark = mp_face_detection.get_key_point(
                        detection, mp_face_detection.FaceKeyPoint.RIGHT_EYE)
                    
                    # 計算左眼和右眼中心點的座標
                    if left_eye_landmark and right_eye_landmark:
                        left_eye_x = int(left_eye_landmark.x * img2.shape[1])
                        left_eye_y = int(left_eye_landmark.y * img2.shape[0])
                        right_eye_x = int(right_eye_landmark.x * img2.shape[1])
                        right_eye_y = int(right_eye_landmark.y * img2.shape[0])
                        eye_center_x = (left_eye_x + right_eye_x) // 2
                        eye_center_y = (left_eye_y + right_eye_y) // 2
                        
                        print("Eye center coordinates (x, y):", eye_center_x, eye_center_y)

            # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
            #將深度影像轉換為彩色深度圖 (depth colormap)
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

            #彩色圖和彩色影像的尺寸資訊
            depth_colormap_dim = depth_colormap.shape
            color_colormap_dim = color_image.shape

            # If depth and color resolutions are different, resize color image to match depth image for display
            #確保深度彩色圖 (depth_colormap) 和彩色影像 (color_image) 的尺寸一致
            if depth_colormap_dim != color_colormap_dim:
                resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)#interpolation參數指定了調整大小時的插值方法
                # images = np.hstack((resized_color_image, depth_colormap))#images變數:包含彩色影像和深度圖的水平組合影像。
            # else:
                # images = np.hstack((color_image, depth_colormap))

            # 獲取中心像素的深度值
            center_pixel_x = depth_frame.width // 2
            center_pixel_y = depth_frame.height // 2
            print(center_pixel_x,center_pixel_y)
            depth_value_center = depth_frame.get_distance(eye_center_x, eye_center_y)
            depth_value_center1 = depth_frame.get_distance(center_pixel_x, center_pixel_y)
            

            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)#創建一個名為 "RealSense" 的視窗，並設置視窗的顯示方式為自動調整大小 
            cv2.circle(color_image, (color_frame.width // 2, color_frame.height // 2), 5, (0, 255, 0), -1)
            cv2.imshow('RealSense', color_image)
            
            center_pixel_x = depth_frame.width // 2
            print("Depth value at center pixel:", depth_value_center*100 ,"cm",eye_center_x,eye_center_y,depth_value_center1*100,img2.shape)
        
            cv2.waitKey(1)

finally:
    # Stop streaming
    pipeline.stop()