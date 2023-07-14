import cv2
import kornia as K
import kornia.feature as KF
import matplotlib.pyplot as plt

import numpy as np
import torch
from kornia_moons.viz import draw_LAF_matches

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

fname1 = 'images/t0.jpg'
fname2 = 'images/t2.jpg'


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

# print(mkpts0.shape)
# print(mkpts1.shape)
mkpts0_resize = mkpts0[::80, :]
mkpts1_resize = mkpts1[::80, :]

Fm, inliers = cv2.findFundamentalMat(mkpts0_resize, mkpts1_resize, cv2.USAC_MAGSAC, 0.5, 0.999, 100000)
inliers = inliers > 0

print(inliers.shape)

draw_LAF_matches(
    KF.laf_from_center_scale_ori(
        torch.from_numpy(mkpts0_resize).view(1, -1, 2),
        torch.ones(mkpts0_resize.shape[0]).view(1, -1, 1, 1),
        torch.ones(mkpts0_resize.shape[0]).view(1, -1, 1),
    ),
    KF.laf_from_center_scale_ori(
        torch.from_numpy(mkpts1_resize).view(1, -1, 2),
        torch.ones(mkpts1_resize.shape[0]).view(1, -1, 1, 1),
        torch.ones(mkpts1_resize.shape[0]).view(1, -1, 1),
    ),
    torch.arange(mkpts0_resize.shape[0]).view(-1, 1).repeat(1, 2),
    K.tensor_to_image(img1),
    K.tensor_to_image(img2),
    inliers,
    draw_dict={
        "inlier_color": (0.2, 1, 0.2),
        "tentative_color": (1.0, 0.5, 1),
        "feature_color": (0.2, 0.5, 1),
        "vertical": False,
    },
    ax = ax,
)

plt.savefig('result.png')
# plt.imshow('result.png',),plt.show()