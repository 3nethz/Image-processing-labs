import matplotlib.pyplot as plt
import numpy as np
import cv2


fig, axes = plt.subplots(3, 2, figsize=(10, 8))
image_path = "PXL_20240621_042323102.jpg"

image = cv2.imread(image_path)

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

R = image_rgb[:, :, 0]
G = image_rgb[:, :, 1]
B = image_rgb[:, :, 2]

grayscale_image = (0.299 * R + 0.587 * G + 0.114 * B).astype(np.uint8)

brightness_increase = int(255 * 0.2)
brightened_image = np.clip(
    grayscale_image.astype(np.int32) + brightness_increase, 0, 255
)
brightened_image = brightened_image.astype(np.uint8)

current_min = grayscale_image.min()
current_max = grayscale_image.max()

new_min = 125
new_max = 175

normalized_image = (grayscale_image - current_min) / (current_max - current_min)
contrast_reduced_image = normalized_image * (new_max - new_min) + new_min
contrast_reduced_image = np.clip(contrast_reduced_image, new_min, new_max)
contrast_reduced_image = (contrast_reduced_image).astype(np.uint8)
four_bpp_image = (contrast_reduced_image / 16).astype(np.uint8) * 16

height = len(grayscale_image)
width = len(grayscale_image[0])

mirror_image = np.zeros((height, width), dtype=np.uint8)

for i in range(height):
    for j in range(width):
        mirror_image[i][width - j - 1] = grayscale_image[i][j]

negative_image = 255 - grayscale_image


axes[0, 0].imshow(grayscale_image, cmap="gray")
axes[0, 0].set_title("Grayscale image")
axes[0, 1].imshow(negative_image, cmap="gray")
axes[0, 1].set_title("Negative image")
axes[1, 0].imshow(brightened_image, cmap="gray")
axes[1, 0].set_title("Brightened image")
axes[1, 1].imshow(contrast_reduced_image, cmap="gray")
axes[1, 1].set_title("Contrast reduced image")
axes[2, 0].imshow(four_bpp_image, cmap="gray")
axes[2, 0].set_title("4bpp image")
axes[2, 1].imshow(mirror_image, cmap="gray")
axes[2, 1].set_title("Mirror image")


plt.tight_layout()

plt.show()
