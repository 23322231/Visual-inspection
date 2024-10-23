import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
#在[0.0, 1.0]範圍內的多個嚴重程度的預先計算矩陣，其中1.0代表最高嚴重程度或雙色覺的情況，而0.0代表無CVD
#紅色色盲
protanomalia_01 = [[0.856167, 0.182038, -0.038205],
                   [0.029342, 0.955115, 0.015544],
                   [-0.002880, -0.001563, 1.004443]]

protanomalia_05 = [[0.458064, 0.679578, -0.137642],
                [0.092785, 0.846313, 0.060902],
                [-0.007494, -0.016807, 1.024301]]

protanomalia_1 = [[0.152286, 1.052583, -0.204868],
                  [0.114503, 0.786281, 0.099216],
                  [-0.003882, -0.048116, 1.051998]]
#綠色色盲
deuteranomaly_01 = [[0.866435, 0.177704, -0.044139],
                    [0.049567, 0.939063, 0.011370],
                    [-0.003453, 0.007233, 0.996220]]

deuteranomaly_05 = [[0.547494, 0.607765, -0.155259],
                    [0.181692, 0.781742, 0.036566],
                    [-0.010410, 0.027275, 0.983136]]

deuteranomaly_1 = [[0.367322, 0.860646, -0.227968],
                   [0.280085, 0.672501, 0.047413],
                   [-0.011820, 0.042940, 0.968881]]
#藍色色盲
tritanomaly_01 = [[0.926670, 0.092514, -0.019184],
                  [0.021191, 0.964503, 0.014306],
                  [0.008437, 0.054813, 0.936750]]

tritanomaly_05 = [[1.017277, 0.027029, -0.044306],
                  [-0.006113, 0.958479, 0.047634],
                  [0.006379, 0.248708, 0.744913]]

tritanomaly_1 = [[1.255528, -0.076749, -0.178779],
                 [-0.078411, 0.930809, 0.147602],
                 [0.004733, 0.691367, 0.303900]]

sim_mats = [[protanomalia_01, protanomalia_05, protanomalia_1],
            [deuteranomaly_01, deuteranomaly_05, deuteranomaly_1],
            [tritanomaly_01, tritanomaly_05, tritanomaly_1]]


'''utils'''
def gammar(rgb, gamma=2.4):
    linear_rgb = np.zeros_like(rgb, dtype=np.float16)
    for i in range(3):
        linear_rgb[:, :, i] = (rgb[:, :, i] / 255) ** gamma

    return linear_rgb


def gamma_correction(s_rgb, gamma=2.4):
    linear_rgb = np.zeros_like(s_rgb, dtype=np.float16)
    for i in range(s_rgb.shape[2]):
        idx = s_rgb[:, :, i] > 0.04045 * 255
        linear_rgb[idx, i] = ((s_rgb[idx, i] / 255 + 0.055) / 1.055) ** gamma
        idx = np.logical_not(idx)
        linear_rgb[idx, i] = s_rgb[idx, i] / 255 / 12.92

    return linear_rgb


def inverse_gamma_correction(linear_rgb, gamma=2.4):
    rgb = np.zeros_like(linear_rgb, dtype=np.float16)
    for i in range(3):
        idx = linear_rgb[:, :, i] <= 0.0031308
        rgb[idx, i] = 255 * 12.92 * linear_rgb[idx, i]
        idx = np.logical_not(idx)
        rgb[idx, i] = 255 * (1.055 * linear_rgb[idx, i]**(1/gamma) - 0.055)

    return np.round(rgb)


def sRGB_from_linearRGB(v):
    if v <= 0.:
        return 0
    if v >= 1.:
        return 255
    if v < 0.0031308:
        return 0.5 + (v * 12.92 * 255)

    return 255 * (pow(v, 1.0 / 2.4) * 1.055 - 0.055)


def clip_array(arr):
    comp_arr = np.ones_like(arr)
    arr = np.maximum(comp_arr * 0, arr)
    arr = np.minimum(comp_arr * 255, arr)

    return arr
'''utils'''

'''convert'''
def linearRGB_from_sRGB(im):
    out = np.zeros_like(im)
    small_mask = im < 0.04045
    large_mask = np.logical_not(small_mask)
    out[small_mask] = im[small_mask] / 12.92
    out[large_mask] = np.power((im[large_mask] + 0.055) / 1.055, 2.4)
    return out


def sRGB_from_linearRGB(im):
    out = np.zeros_like(im)
    # Make sure we're in range, otherwise gamma will go crazy.
    im = np.clip(im, 0., 1.)    #將數組中的元素限制在0-1之間, 防止之後運算溢位
    small_mask = im < 0.0031308
    large_mask = np.logical_not(small_mask)
    out[small_mask] = im[small_mask] * 12.92
    out[large_mask] = np.power(im[large_mask], 1.0 / 2.4) * 1.055 - 0.055
    return out
'''convert'''

'''simulation'''
def find_areas_exp(img_original, img_simulated):
    to_show = False
    res_image = img_original.copy()

    frames_diff_red = cv2.absdiff(img_original[:,:,2], img_simulated[:,:,2])
    #frames_diff_red *= (255 // np.max(frames_diff_red))
    frames_diff_green = cv2.absdiff(img_original[:,:,1], img_simulated[:,:,1])
    #frames_diff_green *= (255 // np.max(frames_diff_green))
    frames_diff_blue = cv2.absdiff(img_original[:,:,1], img_simulated[:,:,1])
    # frames_diff_blue = cv2.absdiff(img_original[:,:,0], img_simulated[:,:,0])
    #frames_diff_blue *= (255 // np.max(frames_diff_blue))
    dst_rg = cv2.addWeighted(frames_diff_red, 1, frames_diff_green, 1, 0)   #根據權重混合兩張圖片
    dst_rgb = cv2.addWeighted(dst_rg, 1, frames_diff_blue, 1, 0)
    # dst_rgb = cv2.addWeighted(dst_rg, 0.67, frames_diff_blue, 0.33, 0)

    ret, thresh1 = cv2.threshold(dst_rgb, 20, 255, cv2.THRESH_BINARY)
    thresh1 = cv2.erode(thresh1, None, iterations=1)
    thresh1 = cv2.dilate(thresh1, None, iterations=1)
    thresh1 = cv2.dilate(thresh1, None, iterations=4)
    thresh1 = cv2.erode(thresh1, None, iterations=4)

    w = 0.6
    dst_f = cv2.addWeighted(res_image, 1-w, cv2.cvtColor(thresh1, cv2.COLOR_GRAY2RGB), w, 0)

    contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    cv2.drawContours(res_image, contours, -1, (1, 244, 225), -1)
    cv2.drawContours(res_image, contours, -1, (0, 0, 0), 1)

    return dst_f

def simulatem(image, cvd, severity):    #色盲種類, 嚴重程度
    im_cv2 = image.copy()
    im_linear_rgb = im_cv2.astype(np.float32) / 255.0   #uint8 -> float32，[0.0, 1.0]
    im_linear_rgb = linearRGB_from_sRGB(im_linear_rgb)
    mat = sim_mats[cvd][severity]
    simul = im_linear_rgb @ np.array(mat).T     #先轉置, 後矩陣相乘@
    im_cvd_float = sRGB_from_linearRGB(simul)

    return (np.clip(im_cvd_float, 0.0, 1.0) * 255.0).astype(np.uint8)

def simulate_color_blindness(image, cb_type, severity):
    # image = cv2.imread(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    im_cv2 = image.copy()

    im_linear_rgb = im_cv2.astype(np.float32) / 255.0   #uint8 -> float32，[0.0, 1.0]
    im_linear_rgb = linearRGB_from_sRGB(im_linear_rgb)
    mat = sim_mats[cb_type][severity]
    simul = im_linear_rgb @ np.array(mat).T     #先轉置, 後矩陣相乘@
    im_cvd_float = sRGB_from_linearRGB(simul)

    return (np.clip(im_cvd_float, 0.0, 1.0) * 255.0).astype(np.uint8)
'''simulation'''