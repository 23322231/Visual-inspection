# 一般鏡頭試mtcnn而已
import cv2
import time
import tensorflow as tf
import detect_face
import numpy as np

# python版本3.10.13

#----tensorflow version check
# if tensorflow.__version__.startswith('1.'):
#     import tensorflow as tf
# else:
#     import tensorflow.compat.v1 as tf
#     tf.disable_v2_behavior()
# print("Tensorflow version: ",tf.__version__)

# 定義視訊初始化函數
def video_init(is_2_write=False,save_path=None):
    # writer = None
    cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
    # 初始化視訊捕獲對象
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)#default 480
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)#default 640

    # width = 480
    # height = 640
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    '''
    ref:https://docs.opencv.org/master/dd/d43/tutorial_py_video_display.html
    FourCC is a 4-byte code used to specify the video codec. 
    The list of available codes can be found in fourcc.org. 
    It is platform dependent. The following codecs work fine for me.
    In Fedora: DIVX, XVID, MJPG, X264, WMV1, WMV2. (XVID is more preferable. MJPG results in high size video. X264 gives very small size video)
    In Windows: DIVX (More to be tested and added)
    In OSX: MJPG (.mp4), DIVX (.avi), X264 (.mkv).
    FourCC code is passed as `cv.VideoWriter_fourcc('M','J','P','G')or cv.VideoWriter_fourcc(*'MJPG')` for MJPG.
    '''

    # 如果需要寫入影像，則初始化影像寫入對象
    # if is_2_write is True:
    #     #fourcc = cv2.VideoWriter_fourcc('x', 'v', 'i', 'd')
    #     #fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
    #     fourcc = cv2.VideoWriter_fourcc(*'divx')
    #     if save_path is None:
    #         save_path = 'demo.avi'
    #     writer = cv2.VideoWriter(save_path, fourcc, 30, (int(width), int(height)))

    # return cap,height,width,writer
    return cap,height,width

# 定義人臉檢測函數
def face_detection_MTCNN(detect_multiple_faces=False):
    
    frame_count = 0
    FPS = "Initialing"
    no_face_str = "No faces detected"

    #初始化視訊流
    # cap, height, width, writer = video_init(is_2_write=False)
    cap, height, width = video_init(is_2_write=False)

    #初始化 MTCNN
    color = (0,255,0)
    minsize = 20  # 人臉的最小尺寸
    threshold = [0.6, 0.7, 0.7]  # 三個步驟的閾值
    factor = 0.709  # scale factor
    with tf.Graph().as_default():
        config = tf.compat.v1.ConfigProto(log_device_placement=True,
                                allow_soft_placement=True,  # 允許當找不到設備時自動轉換成有支援的設備
                                )
        # config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.1
        sess = tf.compat.v1.Session(config=config)
        with sess.as_default():
            pnet, rnet, onet = detect_face.create_mtcnn(sess, None)


    while (cap.isOpened()):

        #----get image
        ret, img = cap.read()
        print(img.shape) #(480, 640, 3)

        if ret is True:
            #影像處理
            img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            print("image shape:",img_rgb.shape)

            #人臉檢測
            t_1 = time.time()
            bounding_boxes, points = detect_face.detect_face(img_rgb, minsize, pnet, rnet, onet, threshold, factor)
            d_t = time.time() - t_1
            print("Time of face detection: ",d_t)

            #邊界框處理
            nrof_faces = bounding_boxes.shape[0]
            if nrof_faces > 0:
                points = np.array(points)
                points = np.transpose(points, [1, 0])
                points = points.astype(np.int16)

                det = bounding_boxes[:, 0:4]
                det_arr = []
                img_size = np.asarray(img.shape)[0:2]
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
                    #det = det.astype(np.int32)
                    cv2.rectangle(img, (det[0],det[1]), (det[2],det[3]), color, 2)

                    #在人臉上繪製 5 個特徵點
                    facial_points = points[i]
                    for j in range(0,5,1):
                        #cv2.circle(影像, 圓心座標, 半徑, 顏色, 線條寬度)
                        cv2.circle(img, (facial_points[j], facial_points[j + 5]), 2, (0, 0, 255), -1, 1)

            #未檢測到人臉
            else:
                cv2.putText(img, no_face_str, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)


            #計算並顯示每秒幀數（FPS）
            if frame_count == 0:
                t_start = time.time()
            frame_count += 1
            if frame_count >= 20:
                FPS = "FPS=%1f" % (frame_count / (time.time() - t_start))
                frame_count = 0

            # cv2.putText(影像, 文字, 座標, 字型, 大小, 顏色, 線條寬度, 線條種類)
            cv2.putText(img, FPS, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

            #----顯示處理後的影像
            cv2.imshow("demo by JohnnyAI", img)

            #----如果需要，將處理後的影像寫入文件
            # if writer is not None:
            #     writer.write(img)

            #----如果按下 'q' 鍵，則退出無限循環
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            print("get image failed")
            break

    #----release
    cap.release()
    # if writer is not None:
    #     writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    face_detection_MTCNN(detect_multiple_faces=True)