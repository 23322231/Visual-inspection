# 深度感測加上UI
import pyrealsense2 as rs
import numpy as np
import cv2
import time
import tensorflow as tf
import detect_face
import tkinter as tk
from tkinter import messagebox
import threading

class RealSenseApp:
    def __init__(self, root):
        self.root = root
        self.root.title("RealSense Camera")
        
        self.start_button = tk.Button(root, text="Start Camera", command=self.start_camera)
        self.start_button.pack(pady=20)
        
        self.fps_label = tk.Label(root, text="FPS: Initializing")
        self.fps_label.pack(pady=5)
        
        self.face_count_label = tk.Label(root, text="Faces detected: 0")
        self.face_count_label.pack(pady=5)
        
        self.distance_label = tk.Label(root, text="Distance from camera to eye center: N/A")
        self.distance_label.pack(pady=5)
        
        self.pipeline = None
        self.pipeline_started = False

    def start_camera(self):
        if self.pipeline_started:
            return
        self.pipeline_started = True
        threading.Thread(target=self.camera_thread).start()

    def update_ui(self, fps, face_count, distance):
        self.fps_label.config(text=f"FPS: {fps:.2f}")
        self.face_count_label.config(text=f"Faces detected: {face_count}")
        self.distance_label.config(text=f"Distance from camera to eye center: {distance:.2f} meters")

    def camera_thread(self):
        # Configure depth and color streams
        self.pipeline = rs.pipeline()
        rs_config = rs.config()

        frame_count = 0
        FPS = 0.0
        no_face_str = "No faces detected"

        # 初始化 MTCNN
        color = (0, 255, 0)
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
        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)  # 將pipeline 封裝成一個包裝器
        pipeline_profile = rs_config.resolve(pipeline_wrapper)  # 獲取相機配置檔案的相關資訊
        device = pipeline_profile.get_device()

        found_rgb = False
        for s in device.sensors:
            if s.get_info(rs.camera_info.name) == 'RGB Camera':
                found_rgb = True
                break
        if not found_rgb:
            messagebox.showerror("Error", "The demo requires Depth camera with Color sensor")
            return

        rs_config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)  # 啟用深度流
        rs_config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)  # 啟用彩色流

        # Start streaming
        self.pipeline.start(rs_config)

        align_to = rs.stream.color
        align = rs.align(align_to)
        try:
            while True:
                eye_center_x = 0
                eye_center_y = 0
                depth_value = 0.0

                # Wait for a coherent pair of frames: depth and color
                frame = self.pipeline.wait_for_frames()
                frames = align.process(frame)
                depth_frame = frames.get_depth_frame()
                color_frame = frames.get_color_frame()
                if not depth_frame or not color_frame:
                    continue

                # Convert images to numpy arrays
                depth_image = np.asanyarray(depth_frame.get_data())
                color_image = np.asanyarray(color_frame.get_data())

                # Apply colormap on depth image
                depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
                depth_colormap_dim = depth_image.shape
                color_colormap_dim = color_image.shape

                # 人臉檢測
                t_1 = time.time()
                bounding_boxes, points = detect_face.detect_face(color_image, minsize, pnet, rnet, onet, threshold, factor)
                d_t = time.time() - t_1

                # 邊界框處理
                nrof_faces = bounding_boxes.shape[0]
                if nrof_faces > 0:
                    points = np.array(points)
                    points = np.transpose(points, [1, 0])
                    points = points.astype(np.int16)
                    det = bounding_boxes[:, 0:4]
                    det_arr = []
                    img_size = np.asarray(color_image.shape)[0:2]
                    detect_multiple_faces = False
                    det_arr.append(np.squeeze(det))
                    det_arr = np.array(det_arr)
                    det_arr = det_arr.astype(np.int16)

                    for i, det in enumerate(det_arr):
                        if len(det) > 0 and len(det) == 4:
                            cv2.rectangle(color_image, (det[0], det[1]), (det[2], det[3]), color, 2)
                        facial_points = points[i]
                        for j in range(0, 5, 1):
                            cv2.circle(color_image, (facial_points[j], facial_points[j + 5]), 2, (0, 0, 255), -1, 1)
                        left_eye_x, left_eye_y = facial_points[0], facial_points[1]
                        right_eye_x, right_eye_y = facial_points[2], facial_points[3]
                        eye_center_x = (right_eye_x + left_eye_x) // 2
                        eye_center_y = (right_eye_y + left_eye_y) // 2

                        if 0 <= eye_center_x < depth_image.shape[1] and 0 <= eye_center_y < depth_image.shape[0]:
                            depth_value = depth_frame.get_distance(eye_center_y, eye_center_x)
                        else:
                            print("Eye center point is out of bounds of depth image.")
                else:
                    cv2.putText(color_image, no_face_str, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

                if frame_count == 0:
                    t_start = time.time()
                frame_count += 1
                if frame_count >= 20:
                    FPS = frame_count / (time.time() - t_start)
                    frame_count = 0

                self.root.after(0, self.update_ui, FPS, nrof_faces, depth_value)

                cv2.putText(color_image, f"FPS: {FPS:.1f}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

                if depth_colormap_dim != color_colormap_dim:
                    resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)

                cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
                combined_image = cv2.hconcat([color_image, depth_colormap])
                cv2.imshow('RealSense', combined_image)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                cv2.waitKey(1)

        finally:
            self.pipeline.stop()


root = tk.Tk()
app = RealSenseApp(root)
root.mainloop()
