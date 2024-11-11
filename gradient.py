# 正式版 計算梯度差異並算邊緣消失比例
import cv2
import numpy as np

# 計算每個通道的梯度
def calculate_gradient(channel):
    grad_x = cv2.Sobel(channel, cv2.CV_64F, 1, 0, ksize=3)  # x方向梯度
    grad_y = cv2.Sobel(channel, cv2.CV_64F, 0, 1, ksize=3)  # y方向梯度
    grad = np.sqrt(grad_x**2 + grad_y**2)  # 合併梯度
    return np.abs(grad)

# 計算梯度差異的絕對值並進行歸一化
def calculate_gradient_difference(img_rgb, img_sim_rgb):
    # 分離通道
    R, G, B = cv2.split(img_rgb)
    R_sim, G_sim, B_sim = cv2.split(img_sim_rgb)

    # 計算原圖與模擬圖的每通道梯度
    grad_R = calculate_gradient(R)
    grad_G = calculate_gradient(G)
    grad_B = calculate_gradient(B)
    grad_R_sim = calculate_gradient(R_sim)
    grad_G_sim = calculate_gradient(G_sim)
    grad_B_sim = calculate_gradient(B_sim)
    # 防止梯度為0時 出現比例100%
    if np.sum(grad_R_sim + grad_G_sim + grad_B_sim) == 0:
        grad_R_sim = grad_R
        grad_G_sim = grad_G
        grad_B_sim = grad_B
    
    # 計算各通道梯度差異(模擬後資訊消失部分)
    r_z = np.zeros((img_rgb.shape[0],img_rgb.shape[1]))
    abs_diff_R = np.maximum(r_z, grad_R - grad_R_sim)
    abs_diff_G = np.maximum(r_z, grad_G - grad_G_sim)
    abs_diff_B = np.maximum(r_z, grad_B - grad_B_sim)
    
    # 計算RGB通道的差異總和
    total_abs_diff = abs_diff_R + abs_diff_G + abs_diff_B
    # 計算原圖的梯度總和
    total_grad = grad_R + grad_G + grad_B
    
    return total_abs_diff, total_grad

# 切分成4x4區塊並計算每個區塊的邊緣消失比例
def process_blocks(img_rgb, img_sim_rgb, num_blocks=4):
    h, w, _ = img_rgb.shape
    block_h, block_w = h // num_blocks, w // num_blocks
    result = np.zeros((h, w))  # 用於存放所有區塊結果
    edge_loss_ratios = np.zeros((num_blocks, num_blocks))  # 用於存放每個區塊的邊緣消失比例

    # 對每個區塊進行計算
    for i in range(num_blocks):
        for j in range(num_blocks):
            # 定義區塊邊界
            y_start, y_end = i * block_h, (i + 1) * block_h
            x_start, x_end = j * block_w, (j + 1) * block_w
            
            # 提取原圖和模擬圖的區塊
            block_img = img_rgb[y_start:y_end, x_start:x_end]
            block_img_sim = img_sim_rgb[y_start:y_end, x_start:x_end]
            
            # 計算區塊的梯度差異
            total_diff, total_grad = calculate_gradient_difference(block_img, block_img_sim)
            # 計算邊緣消失比例 (防止除以零，加入小值 epsilon)
            edge_loss_ratio = (np.sum(total_diff) / (np.sum(total_grad) + 1e-10)) * 100
            edge_loss_ratios[i, j] = edge_loss_ratio
            # 計算結果矩陣
            result[y_start:y_end, x_start:x_end] = edge_loss_ratio

     # 計算區塊的梯度差異
    total_diff_ori, total_grad_ori = calculate_gradient_difference(img_rgb, img_sim_rgb)
    # 計算邊緣消失比例 (防止除以零，加入小值 epsilon)
    edge_loss_ratio_ori = (np.sum(total_diff_ori) / (np.sum(total_grad_ori) + 1e-10)) * 100

    return edge_loss_ratios, edge_loss_ratio_ori


def calculate_edge_loss(img_rgb, img_sim_rgb):
    edge_loss_ratios, edge_loss_ratio_ori = process_blocks(img_rgb, img_sim_rgb, num_blocks=4)
    # 找4x4區塊中邊緣消失比例的最大值
    max_value = np.max(edge_loss_ratios)

    return edge_loss_ratio_ori, max_value
