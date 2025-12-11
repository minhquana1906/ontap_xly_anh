import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def them_nhieu_muoi_tieu(image, prob=0.05):
    output = np.copy(image)
    noise_matrix = np.random.random(image.shape)
    output[noise_matrix < (prob / 2)] = 255
    output[noise_matrix > (1 - prob / 2)] = 0

    return output


def pad_anh_thu_cong(image, pad_width=1):
    h, w = image.shape
    padded_h = h + 2 * pad_width
    padded_w = w + 2 * pad_width
    padded_img = np.zeros((padded_h, padded_w), dtype=int)
    padded_img[pad_width : pad_width + h, pad_width : pad_width + w] = image

    return padded_img


def loc_trung_vi_3x3(image):
    h, w = image.shape
    output = np.zeros((h, w), dtype=np.uint8)
    padded = pad_anh_thu_cong(image, pad_width=1)

    for i in range(h):
        for j in range(w):
            region = padded[i : i + 3, j : j + 3]
            sorted_region = np.sort(region.flatten())
            median_val = sorted_region[4]
            output[i, j] = median_val

    return output


def main():
    img = Image.open("image.jpg").convert("L")
    img_arr = np.array(img)

    noisy_img_arr = them_nhieu_muoi_tieu(img_arr, prob=0.08)  # 8%

    denoised_img_arr = loc_trung_vi_3x3(noisy_img_arr)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    titles = ["Ảnh Gốc", "Ảnh Nhiễu (Salt & Pepper)", "Ảnh Đã Lọc Trung Vị"]
    images = [img_arr, noisy_img_arr, denoised_img_arr]

    for ax, img_data, title in zip(axes, images, titles):
        ax.imshow(img_data, cmap="gray", vmin=0, vmax=255)
        ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
