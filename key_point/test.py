import cv2
import kornia as K
from kornia.io import load_image
import kornia.feature as KF
import matplotlib.pyplot as plt
import numpy as np
import torch
# from kornia_moons.viz import draw_LAF_matches

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

fname1 = "C:\\Users\\lixin\\dataset\\image comparison\\t0.jpg"
fname2 = "C:\\Users\\lixin\\dataset\\image comparison\\t1.jpg"

image1 = cv2.imread(fname1)
image2 = cv2.imread(fname2)

image1 = cv2.resize(image1, (640, 480))
image2 = cv2.resize(image2, (640, 480))


img1 = K.io.load_image(fname1, K.io.ImageLoadType.RGB32)[None, ...]
img2 = K.io.load_image(fname2, K.io.ImageLoadType.RGB32)[None, ...]


img1 = K.geometry.resize(img1, (480, 640), antialias=True)
img2 = K.geometry.resize(img2, (480, 640), antialias=True)

matcher = KF.LoFTR(pretrained="indoor_new")

input_dict = {
    "image0": K.color.rgb_to_grayscale(img1),  # LofTR works on grayscale images only
    "image1": K.color.rgb_to_grayscale(img2),
}

with torch.inference_mode():
    correspondences = matcher(input_dict)

mkpts0 = correspondences["keypoints0"].cpu().numpy()
mkpts1 = correspondences["keypoints1"].cpu().numpy()

#print(mkpts0.shape)
#print(mkpts1.shape)
mkpts0_resize = mkpts0[::30, :]
mkpts1_resize = mkpts1[::30, :]

Fm, inliers = cv2.findFundamentalMat(mkpts0_resize, mkpts1_resize, cv2.USAC_MAGSAC, 0.9, 0.999, 100000)
inliers = inliers > 0

# print(inliers)

match_image = np.zeros((max(image1.shape[0], image2.shape[0]), image1.shape[1] + image2.shape[1], 3), dtype=np.uint8)
match_image[:image1.shape[0], :image1.shape[1]] = image1
match_image[:image2.shape[0], image1.shape[1]:] = image2

for index, match in enumerate(mkpts0_resize):
    if not inliers[index]:
        continue
    kp1 = match
    kp2 = mkpts1_resize[index]
    
    # Calculate the angle between the key points
    angle = np.arctan2(kp2[1] - kp1[1], kp2[0] - kp1[0]) * 180 / np.pi
    
    # Set the line color based on the angle
    if abs(angle) > 25:
        line_color = (255, 0, 0)  # Blue
    else:
        line_color = (0, 0, 255)  # Red
    
    # Draw the line on the match image
    cv2.line(match_image, (int(kp1[0]), int(kp1[1])), (int(kp2[0] + image1.shape[1]), int(kp2[1])), line_color, 1)

# Display the match image
cv2.imshow("Matches", match_image)


cv2.waitKey(0)
cv2.destroyAllWindows()
