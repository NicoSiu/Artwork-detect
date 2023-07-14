from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import imutils
import cv2
import numpy as np

# load
image1_path = "C:\\Users\\lixin\\dataset\\image comparison\\base1.jpg"
image2_path = "C:\\Users\\lixin\\dataset\\image comparison\\move3.jpg"
imageA = cv2.imread(image1_path)
imageA = cv2.resize(imageA, (1280, 720))
imageB = cv2.imread(image2_path)
imageB = cv2.resize(imageB, (1280, 720))
grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)


# hist
x_hist_1 = np.sum(grayA, axis=0)
y_hist_1 = np.sum(grayA, axis=1)
x_hist_1 = x_hist_1.astype(np.float32)
y_hist_1 = y_hist_1.astype(np.float32)

plt.figure()
plt.plot(x_hist_1)
plt.title("X-axis Gray Level Histogram for A")
plt.xlabel("Pixel Value")
plt.ylabel("Frequency")

plt.figure()
plt.plot(y_hist_1)
plt.title("Y-axis Gray Level Histogram for A")
plt.xlabel("Pixel Value")
plt.ylabel("Frequency")


x_hist_2 = np.sum(grayB, axis=0)
y_hist_2 = np.sum(grayB, axis=1)
x_hist_2 = x_hist_2.astype(np.float32)
y_hist_2 = y_hist_2.astype(np.float32)

plt.figure()
plt.plot(x_hist_2)
plt.title("X-axis Gray Level Histogram for B")
plt.xlabel("Pixel Value")
plt.ylabel("Sum")

plt.figure()
plt.plot(y_hist_2)
plt.title("Y-axis Gray Level Histogram for B")
plt.xlabel("Pixel Value")
plt.ylabel("Sum")

#plt.show()


# similarity
Correl_value_x = cv2.compareHist(x_hist_1, x_hist_2, cv2.HISTCMP_CORREL)
print("CORREL value for x: ", Correl_value_x)
Correl_value_y = cv2.compareHist(y_hist_1, y_hist_2, cv2.HISTCMP_CORREL)
print("CORREL value for y: ", Correl_value_y)


CHISQR_x = cv2.compareHist(x_hist_1, x_hist_2, cv2.HISTCMP_CHISQR)
print("CHISQR value for x: ", CHISQR_x)
CHISQR_y = cv2.compareHist(y_hist_1, y_hist_2, cv2.HISTCMP_CHISQR)
print("CHISQR value for y: ", CHISQR_y)


result = cv2.compareHist(x_hist_1, x_hist_2, cv2.HISTCMP_HELLINGER)
print("HELLINGER value for x: ", result)
result = cv2.compareHist(y_hist_1, y_hist_2, cv2.HISTCMP_HELLINGER)
print("HELLINGER value for y: ", result)

# threshold
threshold = 2000000

# text
text = "Please check artwork!"
text_1 = "Nothing special!"
org = (450, 180)
fontFace = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 1
color = (0, 0, 255)
color_1 = (0, 255, 0)

# judge and draw
if CHISQR_x > threshold or CHISQR_y > threshold:
    cv2.putText(imageB, text, org, fontFace, fontScale, color, thickness=2, lineType=cv2.LINE_AA)
else:
    cv2.putText(imageB, text_1, org, fontFace, fontScale, color_1, thickness=2, lineType=cv2.LINE_AA)


# show
cv2.imshow("imageA", imageA)
cv2.imshow("imageB", imageB)
cv2.waitKey(0)
cv2.destroyAllWindows()
