import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def tinh_histogram(image_array):
    hist, bins = np.histogram(image_array.flatten(), bins=256, range=(0, 256))
    return hist


def phan_vung_nguong(image_array, T):
    output_img = np.zeros_like(image_array)
    output_img[image_array >= T] = 255
    return output_img


def main():
    img = Image.open("image.jpg").convert("L")
    img_arr = np.array(img)
    hist = tinh_histogram(img_arr)

    T = 128

    segmented_img = phan_vung_nguong(img_arr, T)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(img_arr, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("Ảnh Gốc (Grayscale)")
    axes[0].axis("off")

    axes[1].plot(hist, color="black")
    axes[1].axvline(T, color="red", linestyle="--", label=f"Ngưỡng T={T}")
    axes[1].set_title("Histogram & Ngưỡng đã chọn")
    axes[1].set_xlabel("Mức xám")
    axes[1].set_ylabel("Số lượng pixel")
    axes[1].legend()
    axes[1].set_xlim([0, 255])

    axes[2].imshow(segmented_img, cmap="gray", vmin=0, vmax=255)
    axes[2].set_title(f"Ảnh Phân Vùng (Binary)\nT = {T}")
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
