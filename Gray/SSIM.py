from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import imutils
import cv2
import numpy as np

image1_path = "C:\\Users\\lixin\\dataset\\image comparison\\5.png"
image2_path = "C:\\Users\\lixin\\dataset\\image comparison\\6.png"
imageA = cv2.imread(image1_path)
imageA = cv2.resize(imageA, (1280, 720))
imageB = cv2.imread(image2_path)
imageB = cv2.resize(imageB, (1280, 720))
grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)
grayA_ = cv2.resize(grayA, (1280, 720))
#cv2.imshow("Img1", grayA_)
# 计算 x 轴和 y 轴上的像素灰度值
x_hist_1 = np.sum(grayA, axis=0)
y_hist_1 = np.sum(grayA, axis=1)
x_hist_1 = x_hist_1.astype(np.float32)
y_hist_1 = y_hist_1.astype(np.float32)

plt.figure()
plt.plot(x_hist_1)
plt.title("X-axis Gray Level Histogram for A")
plt.xlabel("Pixel Value")
plt.ylabel("Frequency")

# 绘制 y 轴上的灰度值直方图
plt.figure()
plt.plot(y_hist_1)
plt.title("Y-axis Gray Level Histogram for A")
plt.xlabel("Pixel Value")
plt.ylabel("Frequency")



grayB_ = cv2.resize(grayB, (1280, 720))
#cv2.imshow("Img2", grayB_)
# 计算 x 轴和 y 轴上的像素灰度值
x_hist_2 = np.sum(grayB, axis=0)
y_hist_2 = np.sum(grayB, axis=1)
x_hist_2 = x_hist_2.astype(np.float32)
y_hist_2 = y_hist_2.astype(np.float32)


plt.figure()
plt.plot(x_hist_2)
plt.title("X-axis Gray Level Histogram for B")
plt.xlabel("Pixel Value")
plt.ylabel("Sum")

# 绘制 y 轴上的灰度值直方图
plt.figure()
plt.plot(y_hist_2)
plt.title("Y-axis Gray Level Histogram for B")
plt.xlabel("Pixel Value")
plt.ylabel("Sum")


plt.show()


'''''
# compute the Structural Similarity Index (SSIM) between the two
# images, ensuring that the difference image is returned
(score, diff) = ssim(grayA, grayB, full=True)
diff = (diff * 255).astype("uint8")
print("SSIM: {}".format(score))

# threshold the difference image, followed by finding contours to
# obtain the regions of the two input images that differ
thresh = cv2.threshold(diff, 0, 255,
	cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

# loop over the contours

print(diff)
filtered_cnts = []
for c in cnts:
    (x, y, w, h) = cv2.boundingRect(c)
    area = w * h

    # 计算差异区域的像素值方差
    variance = np.var(diff[y:y+h, x:x+w])

    # 根据阈值过滤差异区域
    region = (diff[y:y + h, x:x + w])
    sum_gray = int(region.sum())
    unit = sum_gray/area
    threshold = 170  # 设置方差阈值
    #(threshold)
    if unit < threshold and area > 500:
        filtered_cnts.append(c)

for c in filtered_cnts:
	# compute the bounding box of the contour and then draw the
	# bounding box on both input images to represent where the two
	# images differ
	(x, y, w, h) = cv2.boundingRect(c)
	cv2.rectangle(imageA, (x, y), (x + w, y + h), (0, 255, 0), 1)
	cv2.rectangle(imageB, (x, y), (x + w, y + h), (0, 0, 255), 1)


# 显示输出图片
resized_image1 = cv2.resize(imageA, (1280, 720))
cv2.imshow("Image1", resized_image1)
resized_image2 = cv2.resize(imageB, (1280, 720))
cv2.imshow("Image2", resized_image2)

resized_image3 = cv2.resize(diff, (1280, 720))
cv2.imshow("Diff", resized_image3)

resized_image4 = cv2.resize(thresh, (1000, 750))
cv2.imshow("Thresh", resized_image4)

cv2.waitKey(0)
'''''