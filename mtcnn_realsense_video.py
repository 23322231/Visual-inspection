# 當初寫的自己錄影片當測茲的版本
import pyrealsense2 as rs
import numpy as np
import cv2
import time
import tensorflow as tf
import detect_face

# 初始化 MTCNN
color = (0, 255, 0)
minsize = 20  # 人臉的最小尺寸
threshold = [0.6, 0.7, 0.7]  # 三個步驟的閾值
no_face_str = "No faces detected"
factor = 0.709  # scale factor
with tf.Graph().as_default():
    config = tf.compat.v1.ConfigProto(log_device_placement=True,
                                      allow_soft_placement=True)  # 允許當找不到設備時自動轉換成有支援的設備
    config.gpu_options.per_process_gpu_memory_fraction = 0.1
    sess = tf.compat.v1.Session(config=config)
    with sess.as_default():
        pnet, rnet, onet = detect_face.create_mtcnn(sess, None)

# 讀取影片
video_path = 'testdata/test1.mp4'  
playback = rs.playback(video_path)
# cap = cv2.VideoCapture(video_path)
detect_multiple_faces = False

# if not cap.isOpened():
#     print("Error: Unable to open video file.")
#     exit()

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_device_from_file(video_path)


# config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
pipeline.start(config)

try:
    while True:
        # ret, frame = cap.read()  # 讀取影片中的一幀
        # print(type(frame))
        # if not ret:
        #     print("Error: Unable to read frame.")
        #     break

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        print(type(frames))
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

        # 人臉檢測
        t_1 = time.time()
        bounding_boxes, points = detect_face.detect_face(color_image, minsize, pnet, rnet, onet, threshold, factor)
        d_t = time.time() - t_1

        #邊界框處理
        nrof_faces = bounding_boxes.shape[0]
        if nrof_faces > 0:
            points = np.array(points)
            points = np.transpose(points, [1, 0])
            points = points.astype(np.int16)

            det = bounding_boxes[:, 0:4]
            det_arr = []
            img_size = np.asarray(color_image.shape)[0:2]
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
                cv2.rectangle(color_image, (det[0],det[1]), (det[2],det[3]), color, 2)#在原始影像上繪製一個矩形

                # 確保人臉位置不會超出深度圖像的範圍
                x1, y1, x2, y2 = det[0], det[1], det[2], det[3]
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(depth_image.shape[1] - 1, x2)
                y2 = min(depth_image.shape[0] - 1, y2)

                # 如果人臉位置超出深度圖像範圍，則跳過深度偵測的部分
                if x1 >= depth_image.shape[1] or y1 >= depth_image.shape[0] or x2 < 0 or y2 < 0:
                    continue

                # 在人臉上繪製 5 個特徵點
                facial_points = points[i]
                for j in range(0, 5, 1):
                    cv2.circle(color_image, (facial_points[j], facial_points[j + 5]), 2, (0, 0, 255), -1, 1)

                # 在人臉中間繪製深度值
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                depth_value = depth_image[center_y, center_x]
                cv2.putText(color_image, f'Depth: {depth_value}', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 未檢測到人臉
        else:
            cv2.putText(color_image, no_face_str, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        # 顯示影片和深度圖
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        cv2.imshow('Video with Face Detection', color_image)
        cv2.imshow('Depth', depth_colormap)

        # 如果按下 'q' 鍵，則退出無限循環
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    # cap.release()
    cv2.destroyAllWindows()
