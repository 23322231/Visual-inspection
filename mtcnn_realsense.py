# 這是最終成功版!!!!
# 測得視窗中間的距離值
import pyrealsense2 as rs
import numpy as np
import cv2
import time
import tensorflow as tf
import detect_face



# Configure depth and color streams
pipeline = rs.pipeline()
rs_config = rs.config()

frame_count = 0
FPS = "Initialing"
no_face_str = "No faces detected"

#初始化 MTCNN
color = (0,255,0)
minsize = 20  # 人臉的最小尺寸
threshold = [0.6, 0.7, 0.7]  # 三個步驟的閾值
factor = 0.709  # scale factor
with tf.Graph().as_default():
    config = tf.compat.v1.ConfigProto(log_device_placement=True,
                            allow_soft_placement=True,  # 允許當找不到設備時自動轉換成有支援的設備
                            )

    config.gpu_options.per_process_gpu_memory_fraction = 0.5
    sess = tf.compat.v1.Session(config=config)
    with sess.as_default():
        pnet, rnet, onet = detect_face.create_mtcnn(sess, None)

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)#將pipeline 封裝成一個包裝器       
pipeline_profile = rs_config.resolve(pipeline_wrapper)#獲取相機配置檔案的相關資訊
device = pipeline_profile.get_device()


found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

rs_config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)#啟用深度流，設置解析度為 640x480，像素格式為 z16（16 位深度值），幀率為 30 幀/秒
rs_config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)#啟用彩色流，設置解析度為 960x540，像素格式為 bgr8（8 位 RGB 彩色），幀率為 30 幀/秒


# Start streaming
pipeline.start(rs_config)

align_to=rs.stream.color
align=rs.align(align_to)
try:
    while True:
        eye_center_x=0
        eye_center_y=0
        depth_value=0

        # Wait for a coherent pair of frames: depth and color
        frame = pipeline.wait_for_frames() #等待串流 pipeline 中的一組幀。這些幀通常包括深度影像、彩色影像等
        frames=align.process(frame)
        # print(1)
        # print(type(frames))
        # print((frames.size()))
        depth_frame = frames.get_depth_frame()#返回深度影像幀。
        color_frame = frames.get_color_frame()#返回彩色影像幀。
        if not depth_frame or not color_frame:#檢查是否成功獲取了深度影像和彩色影像幀
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())#獲取深度影像的資料並轉換為 NumPy 陣列
        color_image = np.asanyarray(color_frame.get_data())#獲取彩色影像的資料並轉換為 NumPy 陣列
        
        
        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        #將深度影像轉換為彩色深度圖 (depth colormap)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        #彩色圖和彩色影像的尺寸資訊
        depth_colormap_dim = depth_image.shape
        color_colormap_dim = color_image.shape
        # print(depth_colormap_dim)
        # print(color_colormap_dim)
        
        # #人臉檢測
        t_1 = time.time()
        # img：待檢測的彩色影像。minsize：人臉的最小尺寸。pnet、rnet、onet：MTCNN 模型的三個子網絡。threshold：三個步驟的閾值。factor：縮放因子。
        # bounding_boxes：檢測到的人臉的邊界框座標，是一個 Numpy 陣列。points：檢測到的人臉的特徵點座標，也是一個 Numpy 陣列。
        bounding_boxes, points = detect_face.detect_face(color_image, minsize, pnet, rnet, onet, threshold, factor)
        d_t = time.time() - t_1

        #邊界框處理
        nrof_faces = bounding_boxes.shape[0]#計算檢測到的人臉數量
        if nrof_faces > 0:
            points = np.array(points)
            points = np.transpose(points, [1, 0])#轉置，將關鍵點的 x 和 y 座標分開
            points = points.astype(np.int16)

            det = bounding_boxes[:, 0:4]#(左上角 x, 左上角 y, 右下角 x, 右下角 y)
            det_arr = []
            img_size = np.asarray(color_image.shape)[0:2]#得到圖像的高度和寬度
            detect_multiple_faces=False#處理多個檢測到的人臉或僅處理其中的一個
            
            # if nrof_faces > 1:
            #     if detect_multiple_faces:
            #         for i in range(nrof_faces):
            #             det_arr.append(np.squeeze(det[i]))#將邊界框添加到 det_arr 列表中
            #     else:
            #         bounding_box_size = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
            #         img_center = img_size / 2
            #         offsets = np.vstack(
            #             [(det[:, 0] + det[:, 2]) / 2 - img_center[1], (det[:, 1] + det[:, 3]) / 2 - img_center[0]])
            #         offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
            #         index = np.argmax(
            #             bounding_box_size - offset_dist_squared * 2.0)  # some extra weight on the centering
            #         det_arr.append(det[index, :])
            # else:
            
            det_arr.append(np.squeeze(det))#det_arr 中的每個元素都是一個表示邊界框的一維陣列

            det_arr = np.array(det_arr)
            det_arr = det_arr.astype(np.int16)

            for i, det in enumerate(det_arr):#遍歷 det_arr 中的每個邊界框
                if len(det) > 0  and len(det) == 4:
                    cv2.rectangle(color_image, (det[0],det[1]), (det[2],det[3]), color, 2)#在原始影像上繪製一個矩形

                #在人臉上繪製 5 個特徵點
                facial_points = points[i]
                for j in range(0,5,1):
                    #cv2.circle(影像, 圓心座標, 半徑, 顏色, 線條寬度)
                    cv2.circle(color_image, (facial_points[j], facial_points[j + 5]), 2, (0, 0, 255), -1, 1)

                # 取出左右眼的位置座標
                left_eye_x, left_eye_y = facial_points[0], facial_points[1]
                right_eye_x, right_eye_y = facial_points[2], facial_points[3]
                print("Left eye coordinates:", (left_eye_x, left_eye_y))
                print("Right eye coordinates:", (right_eye_x, right_eye_y))
                print("eye_center_x",(right_eye_x + left_eye_x)//2)
                print("eye_center_y",(right_eye_y + left_eye_y)//2)
                eye_center_x=(right_eye_x + left_eye_x)//2
                eye_center_y=(right_eye_y + left_eye_y)//2

                if 0 <= eye_center_x < depth_image.shape[1] and 0 <= eye_center_y < depth_image.shape[0]:
                    # 從深度影像中獲取眼睛中心點的深度值
                    depth_value = depth_frame.get_distance(eye_center_y, eye_center_x)
                    # 眼睛中心點與相機的距離
                    print("Distance from camera to eye center (in meters):", depth_value)
                else:
                    print("Eye center point is out of bounds of depth image.")
        #未檢測到人臉
        else:
            cv2.putText(color_image, no_face_str, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        #計算並顯示每秒幀數（FPS）
        if frame_count == 0:
            t_start = time.time()
        frame_count += 1
        if frame_count >= 20:
            FPS = "FPS=%1f" % (frame_count / (time.time() - t_start))
            frame_count = 0

        # cv2.putText(影像, 文字, 座標, 字型, 大小, 顏色, 線條寬度, 線條種類)
        cv2.putText(color_image, FPS, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
        
        #將深度影像轉換為彩色深度圖 (depth colormap)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        # If depth and color resolutions are different, resize color image to match depth image for display
        #確保深度彩色圖 (depth_colormap) 和彩色影像 (color_image) 的尺寸一致
        if depth_colormap_dim != color_colormap_dim:
            resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)#interpolation參數指定了調整大小時的插值方法
        
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        combined_image = cv2.hconcat([color_image, depth_colormap])
        cv2.imshow('RealSense', combined_image)

        #----如果按下 'q' 鍵，則退出無限循環
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        cv2.waitKey(1)

finally:
    # Stop streaming
    pipeline.stop()