import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def tinh_histogram(image_array):
    hist, _ = np.histogram(image_array.flatten(), bins=256, range=(0, 256))
    return hist


def phan_vung_anh(image_array, t1, t2):
    output_img = np.zeros_like(image_array)

    mask_toi = image_array < t1
    mask_tb = (image_array >= t1) & (image_array < t2)
    mask_sang = image_array >= t2

    output_img[mask_toi] = 0
    output_img[mask_tb] = 128
    output_img[mask_sang] = 255

    # Cách 2 (ngắn gọn hơn): Dùng np.select
    # conditions = [image_array < t1, (image_array >= t1) & (image_array < t2), image_array >= t2]
    # choices = [0, 128, 255]
    # output_img = np.select(conditions, choices)

    return output_img


def main():
    img = Image.open("image.jpg").convert("L")
    img_arr = np.array(img)

    hist = tinh_histogram(img_arr)

    T1 = 85  # Ngưỡng tối
    T2 = 170  # Ngưỡng sáng

    segmented_img = phan_vung_anh(img_arr, T1, T2)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(img_arr, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("Ảnh Gốc")
    axes[0].axis("off")

    axes[1].plot(hist, color="black")
    axes[1].axvline(T1, color="red", linestyle="--", label=f"T1={T1}")
    axes[1].axvline(T2, color="blue", linestyle="--", label=f"T2={T2}")
    axes[1].set_title("Histogram & Ngưỡng phân vùng")
    axes[1].legend()
    axes[1].set_xlim([0, 255])

    axes[2].imshow(segmented_img, cmap="gray", vmin=0, vmax=255)
    axes[2].set_title("Ảnh Phân Vùng (3 Mức)")
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
