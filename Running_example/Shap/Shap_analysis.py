from S1_Model import S1_Model
import cv2
import numpy


from skimage.segmentation import slic
import matplotlib.pyplot as plt
import numpy as np
import shap

# load model data
"""
This provides an example for shap for S1_Model(object detection)
Load other models here to explain other models 
"""
s1_model = S1_Model()
model = s1_model.model

# load an image
img_0 = cv2.imread(r"Shap/data_s1/1.png")  # Image to be detected
gray = cv2.cvtColor(img_0, cv2.COLOR_RGB2GRAY)
img = numpy.zeros_like(img_0)
for i in range(3):
    img[:, :, i] = gray  # Convert to grayscale for inference


# segment the image so we don't have to explain every pixel
segments_slic = slic(img, n_segments=150, compactness=0.1, sigma=1)
h, w = segments_slic.shape
new_map = np.zeros_like(segments_slic)


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

# run the explainer
explainer = shap.KernelExplainer(f, np.zeros((1, 200)))
shap_values = explainer.shap_values(np.ones((1, 200)), nsamples=1000)  # runs model 1000 times

explainer = shap.explainers.Permutation(model, img)
shap_value_single = explainer(img)


# plot our explanations
def fill_segmentation(values, segmentation):
    out = np.zeros(segmentation.shape)
    for i in range(len(values[0])):
        out[segmentation == i] = values[0, i]
    return out


img_0 = cv2.cvtColor(img_0, cv2.COLOR_BGR2RGB)
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))

axes[0].imshow(img_0)
axes[0].axis('off')
max_val = np.max([np.max(np.abs(shap_values[i][:, :-1])) for i in range(len(shap_values))])

axes[1].imshow(img_0)
m = fill_segmentation(shap_values[0], segments_slic)
im = axes[1].imshow(m, cmap="bwr", vmin=0, vmax=max_val, alpha=0.5)
axes[1].axis('off')

cb = fig.colorbar(im, ax=axes.ravel().tolist(), label="SHAP value", orientation="horizontal", aspect=60)
cb.outline.set_visible(False)
plt.savefig(r"Shap/data_s1/1_test.png", dpi=1200)


