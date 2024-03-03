import cv2
import numpy as np
import pyrealsense2 as rs

# 初始化 RealSense 攝像頭設備
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
profile = pipeline.start(config)

align_to = rs.align(rs.stream.color)

# 網絡初始化，這裡的 "MobileNetSSD_deploy.prototxt" 和 "MobileNetSSD_deploy.caffemodel"
# 文件需要替換成相應的 Python 模型文件
net = cv2.dnn.readNetFromCaffe("MobileNetSSD_deploy.prototxt", "MobileNetSSD_deploy.caffemodel")

inWidth, inHeight = 300, 300
WHRatio = inWidth / float(inHeight)
inScaleFactor = 0.007843
meanVal = 127.5
classNames = ["background", "aeroplane", "bicycle", "bird", "boat",
              "bottle", "bus", "car", "cat", "chair",
              "cow", "diningtable", "dog", "horse",
              "motorbike", "person", "pottedplant",
              "sheep", "sofa", "train", "tvmonitor"]

try:
    while True:
        # Wait for the next set of frames
        frames = pipeline.wait_for_frames()
        # Make sure the frames are spatially aligned
        aligned_frames = align_to.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()

        # If we only received new depth frame,
        # but the color did not update, continue
        if not color_frame or not depth_frame:
            continue

        # Convert RealSense frame to OpenCV matrix
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        # Crop both color and depth frames
        height, width, _ = color_image.shape
        crop_size = (int(height * WHRatio), height) if width / float(height) > WHRatio else (width, int(width / WHRatio))
        crop = ((width - crop_size[0]) // 2, (height - crop_size[1]) // 2, (width + crop_size[0]) // 2, (height + crop_size[1]) // 2)

        color_image = color_image[crop[1]:crop[3], crop[0]:crop[2]]
        depth_image = depth_image[crop[1]:crop[3], crop[0]:crop[2]]

        input_blob = cv2.dnn.blobFromImage(color_image, inScaleFactor, (inWidth, inHeight), meanVal, False)
        net.setInput(input_blob, "data")
        detection = net.forward("detection_out")

        detection_mat = detection.reshape(detection.shape[2], detection.shape[3])

        confidence_threshold = 0.8
        for i in range(detection_mat.shape[0]):
            confidence = detection_mat[i, 2]

            if confidence > confidence_threshold:
                object_class = int(detection_mat[i, 1])
                x_left_bottom = int(detection_mat[i, 3] * color_image.shape[1])
                y_left_bottom = int(detection_mat[i, 4] * color_image.shape[0])
                x_right_top = int(detection_mat[i, 5] * color_image.shape[1])
                y_right_top = int(detection_mat[i, 6] * color_image.shape[0])

                object_rect = (x_left_bottom, y_left_bottom, x_right_top - x_left_bottom, y_right_top - y_left_bottom)
                object_rect = (max(0, object_rect[0]), max(0, object_rect[1]), min(object_rect[2], depth_image.shape[1]), min(object_rect[3], depth_image.shape[0]))

                # Calculate mean depth inside the detection region
                # This is a very naive way to estimate objects depth
                # but it is intended to demonstrate how one might
                # use depth data in general
                depth_roi = depth_image[object_rect[1]:object_rect[1] + object_rect[3], object_rect[0]:object_rect[0] + object_rect[2]]
                m = np.mean(depth_roi)

                label_text = f"{classNames[object_class]} {m:.2f} meters away"
                cv2.rectangle(color_image, (object_rect[0], object_rect[1]), (object_rect[0] + object_rect[2], object_rect[1] + object_rect[3]), (0, 255, 0), 2)
                cv2.putText(color_image, label_text, (object_rect[0], object_rect[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow("Display Image", color_image)
        if cv2.waitKey(1) >= 0:
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
