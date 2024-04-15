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

    config.gpu_options.per_process_gpu_memory_fraction = 0.1
    sess = tf.compat.v1.Session(config=config)
    with sess.as_default():
        pnet, rnet, onet = detect_face.create_mtcnn(sess, None)

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = rs_config.resolve(pipeline_wrapper)
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

rs_config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# if device_product_line == 'L500':
rs_config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
# else:
    # config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(rs_config)

try:
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
        img=np.asanyarray(color_frame.get_data())

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        #將深度影像轉換為彩色深度圖 (depth colormap)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        #彩色圖和彩色影像的尺寸資訊
        depth_colormap_dim = depth_colormap.shape
        color_colormap_dim = color_image.shape
        
        # #人臉檢測
        t_1 = time.time()
        bounding_boxes, points = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
        d_t = time.time() - t_1

        #邊界框處理
        nrof_faces = bounding_boxes.shape[0]
        if nrof_faces > 0:
            points = np.array(points)
            points = np.transpose(points, [1, 0])
            points = points.astype(np.int16)

            left_eye_x = points[0][0]  # 左眼 x 座標
            left_eye_y = points[0][1]  # 左眼 y 座標
            right_eye_x = points[0][2]  # 右眼 x 座標
            right_eye_y = points[0][3]  # 右眼 y 座標
            eye_center_x = (left_eye_x + right_eye_x) // 2
            eye_center_y = (left_eye_y + right_eye_y) // 2
            print("Eye center coordinates (x, y):", eye_center_x, eye_center_y)

            det = bounding_boxes[:, 0:4]
            det_arr = []
            img_size = np.asarray(img.shape)[0:2]
            detect_multiple_faces=False
            
            if nrof_faces > 1:
                if detect_multiple_faces:
                    for i in range(nrof_faces):
                        det_arr.append(np.squeeze(det[i]))
                else:
                    bounding_box_size = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
                    img_center = img_size / 2
                    offsets = np.vstack(
                        [(det[:, 0] + det[:, 2]) / 2 - img_center[1], (det[:, 1] + det[:, 3]) / 2 - img_center[0]])
                    offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
                    index = np.argmax(
                        bounding_box_size - offset_dist_squared * 2.0)  # some extra weight on the centering
                    det_arr.append(det[index, :])
            else:
                det_arr.append(np.squeeze(det))

            det_arr = np.array(det_arr)
            det_arr = det_arr.astype(np.int16)

            for i, det in enumerate(det_arr):
                cv2.rectangle(img, (det[0],det[1]), (det[2],det[3]), color, 2)

                #在人臉上繪製 5 個特徵點
                facial_points = points[i]
                for j in range(0,5,1):
                    #cv2.circle(影像, 圓心座標, 半徑, 顏色, 線條寬度)
                    cv2.circle(img, (facial_points[j], facial_points[j + 5]), 2, (0, 0, 255), -1, 1)
        
        #將深度影像轉換為彩色深度圖 (depth colormap)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        #彩色圖和彩色影像的尺寸資訊
        depth_colormap_dim = depth_colormap.shape
        color_colormap_dim = color_image.shape

        # If depth and color resolutions are different, resize color image to match depth image for display
        #確保深度彩色圖 (depth_colormap) 和彩色影像 (color_image) 的尺寸一致
        if depth_colormap_dim != color_colormap_dim:
            resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)#interpolation參數指定了調整大小時的插值方法
        #     images = np.hstack((resized_color_image, depth_colormap))#images變數:包含彩色影像和深度圖的水平組合影像。
        # else:
        #     images = np.hstack((color_image, depth_colormap))

        #計算並顯示每秒幀數（FPS）
            if frame_count == 0:
                t_start = time.time()
            frame_count += 1
            if frame_count >= 20:
                FPS = "FPS=%1f" % (frame_count / (time.time() - t_start))
                frame_count = 0

            # cv2.putText(影像, 文字, 座標, 字型, 大小, 顏色, 線條寬度, 線條種類)
            cv2.putText(img, FPS, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
        
        # 獲取中心像素的深度值
        center_pixel_x = depth_frame.width // 2
        center_pixel_y = depth_frame.height // 2
        print(center_pixel_x,center_pixel_y)
        depth_value_center = depth_frame.get_distance(eye_center_x, eye_center_y)
        depth_value_center1 = depth_frame.get_distance(center_pixel_x, center_pixel_y)
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)


        cv2.circle(color_image, (color_frame.width // 2, color_frame.height // 2), 5, (0, 255, 0), -1)
        cv2.imshow('RealSense', color_image)
        center_pixel_x = depth_frame.width // 2
        print("Depth value at center pixel:", depth_value_center*100 ,"cm",eye_center_x,eye_center_y,depth_value_center1*100,img.shape)
        # print("Depth value at center pixel:", depth_value_center ,"m",center_pixel_x,center_pixel_y,depth_colormap.shape)
        cv2.waitKey(1)

finally:
    # Stop streaming
    pipeline.stop()