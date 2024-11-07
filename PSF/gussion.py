import cv2

# 讀取圖片
image = cv2.imread('C:\\xampp\\htdocs\\Visual-inspection\\PSF\\letter_e.png')  # 請將 'input_image.jpg' 替換為您的圖片路徑

# 應用高斯模糊 (kernel size 可調整)
blurred_image = cv2.GaussianBlur(image, (25, 25), 0)

# 儲存模糊後的圖片
cv2.imwrite('output_blurred_image.jpg', blurred_image)  # 'output_blurred_image.jpg' 是儲存的檔名
print("圖片已成功模糊並儲存為 'output_blurred_image.jpg'")
