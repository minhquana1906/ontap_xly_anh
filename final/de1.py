import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def tinh_histogram(image_array):
    hist = np.zeros(256, dtype=int)
    for pixel in image_array.flatten():
        hist[pixel] += 1
    return hist


def tinh_cdf(pdf):
    cdf = np.zeros(256, dtype=float)
    cdf[0] = pdf[0]
    for i in range(1, 256):
        cdf[i] = cdf[i - 1] + pdf[i]
    return cdf


def main():
    img = Image.open("image.jpg").convert("L")

    img_arr = np.array(img)
    total_pixels = img_arr.size

    hist = tinh_histogram(img_arr)

    pdf = hist / total_pixels

    cdf = tinh_cdf(pdf)

    cdf_max = cdf.max()  # Giá trị này sẽ xấp xỉ 1
    new_levels = np.round((cdf / cdf_max) * 255).astype(np.uint8)

    equalized_img_arr = new_levels[img_arr]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Ảnh gốc
    axes[0].imshow(img_arr, cmap="gray", vmin=0, vmax=255)
    # axes[0].set_title("Ảnh Gốc")
    axes[0].axis("off")

    # Ảnh sau cân bằng
    axes[1].imshow(equalized_img_arr, cmap="gray", vmin=0, vmax=255)
    # axes[1].set_title("Ảnh Cân Bằng Histogram")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
