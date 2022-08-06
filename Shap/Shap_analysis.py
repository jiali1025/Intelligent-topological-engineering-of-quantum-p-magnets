import os, time, gc, pickle, itertools, math
from typing import List
import numpy as np
import pandas as pd
import shap
from S3_Model import S3_Model
import cv2
import numpy

import requests
from skimage.segmentation import slic
import matplotlib.pyplot as plt
import numpy as np
import shap

# load model data
s3_model = S3_Model()
model = s3_model.model

# load an image
img_0 = cv2.imread(r"Shap/data_s1/1.png")  # Image to be detected
gray = cv2.cvtColor(img_0, cv2.COLOR_RGB2GRAY)

img = numpy.zeros_like(img_0)
for i in range(3):
    img[:, :, i] = gray  # Convert to grayscale for inference

# img = cv2.cvtColor(img_0, cv2.COLOR_RGB2GRAY)

# segment the image so we don't have to explain every pixel
segments_slic = slic(img, n_segments=150, compactness=0.1, sigma=1)

h, w = segments_slic.shape

new_map = np.zeros_like(segments_slic)


def is_edge(i, j):
    val = segments_slic[i, j]
    val_l = segments_slic[i - 1, j]
    val_r = segments_slic[i + 1, j]
    val_u = segments_slic[i, j + 1]
    val_d = segments_slic[i, j - 1]

    s = val == [val_l, val_r, val_d, val_u]
    c = np.any(s == False)
    return c


# define a function that depends on a binary mask representing if an image region is hidden
def mask_image(zs, segmentation, image, background=None):
    if background is None:
        background = image.mean((0, 1))
    out = np.zeros((zs.shape[0], image.shape[0], image.shape[1], image.shape[2]))
    for i in range(zs.shape[0]):
        out[i, :, :, :] = image
        for j in range(zs.shape[1]):
            if zs[i, j] == 0:
                out[i][segmentation == j, :] = background
    return out


def f(z):
    return model(mask_image(z, segments_slic, img, None))


explainer = shap.KernelExplainer(f, np.zeros((1, 200)))
shap_values = explainer.shap_values(np.ones((1, 200)), nsamples=600)  # runs VGG16 1000 times


# ap = model(img)

# explainer = shap.explainers.Permutation(model, img_gray)
# shap_value_single = explainer(img_gray)

def fill_segmentation(values, segmentation):
    out = np.zeros(segmentation.shape)
    for i in range(len(values[0])):
        out[segmentation == i] = values[0, i]
    return out


img_0 = cv2.cvtColor(img_0, cv2.COLOR_BGR2RGB)

# plot our explanations
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
# inds = top_preds[0]
axes[0].imshow(img_0)
axes[0].axis('off')
max_val = np.max([np.max(np.abs(shap_values[i][:, :-1])) for i in range(len(shap_values))])

# for i in range(3):
m = fill_segmentation(shap_values[0], segments_slic)
# axes[i+1].set_title(feature_names[str(inds[i])][1])


axes[1].imshow(img_0)
im = axes[1].imshow(m, cmap="bwr", vmin=0, vmax=max_val, alpha=0.5)
axes[1].axis('off')

cb = fig.colorbar(im, ax=axes.ravel().tolist(), label="SHAP value", orientation="horizontal", aspect=60)
cb.outline.set_visible(False)
plt.savefig(r"Shap/data_s1/1_test.png", dpi=1200)

# --------------------------------------- show segments --------------------------------------------- #
# for i in range(h - 2):
#     for j in range(w - 2):
#         if is_edge(i + 1, j + 1):
#             new_map[i + 1, j + 1] = 1

# img_0 = cv2.cvtColor(img_0, cv2.COLOR_BGR2RGB)


# img_0[:, :, 0] = img_0[:, :, 0] + new_map * 200
# img_0[:, :, 1] = img_0[:, :, 1] + new_map * 200

# img_0[new_map==1] = [255,165,0]

# plt.imshow(img_0)
# plt.imshow(new_map, cmap="Oranges", alpha=0.5, vmin=0, vmax=2)
# plt.show()
# plt.savefig(r"/home/pengfei/projects/neural_ode/STM/Explainable/data_s1/13_seg.png", dpi=600)
