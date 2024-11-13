from PIL import Image

# 讀取圖片檔案
image = Image.open("C:\\Users\\user\\Documents\\GitHub\\Visual-inspection\\static\\images\\home-icon.png").convert("RGBA")  # 確保是 RGBA 格式

# 創建一個新的圖片，將所有非透明背景的像素改為白色
pixels = image.load()
for y in range(image.height):
    for x in range(image.width):
        r, g, b, a = pixels[x, y]
        if a != 0:  # 檢查是否為非透明像素
            pixels[x, y] = (255, 255, 255, 255)  # 將像素設為白色（不透明）

# 儲存新的圖片檔案
image.save("home-icon.png")
