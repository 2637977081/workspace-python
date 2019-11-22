import cv2

input_image_path='..\\resources\\images\\yaojing.png'
# input_image_path='..\\resources\\images\\28625390.jpg'

img_rgb = cv2.imread(input_image_path)
#转化为灰度图
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
# 对原图进行模糊化
img_gray = cv2.medianBlur(img_gray, 5)
# 二值化操作
img_edge = cv2.adaptiveThreshold(img_gray, 255,
                                 cv2.ADAPTIVE_THRESH_MEAN_C,
                                 cv2.THRESH_BINARY, blockSize=3, C=2)


output_image_path='..\\resources\\images\\yaojing1.png'
cv2.imwrite(output_image_path, img_edge)
