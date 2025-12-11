import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def them_nhieu_muoi_tieu(image, prob=0.05):
    """
    Tự viết hàm thêm nhiễu Salt & Pepper.
    prob: Tỉ lệ nhiễu (0.05 tương ứng 5%)
    """
    output = np.copy(image)
    # Tạo ma trận ngẫu nhiên cùng kích thước
    noise_matrix = np.random.random(image.shape)

    # Thêm muối (Salt - trắng - 255): nửa số lượng prob
    output[noise_matrix < (prob / 2)] = 255

    # Thêm tiêu (Pepper - đen - 0): nửa số lượng prob còn lại
    output[noise_matrix > (1 - prob / 2)] = 0

    return output


def pad_anh_thu_cong(image, pad_width=1):
    """
    Tự viết hàm Zero-padding.
    Tạo ma trận 0 lớn hơn và đặt ảnh gốc vào giữa.
    """
    h, w = image.shape
    # Kích thước mới = cũ + 2 * lề
    padded_h = h + 2 * pad_width
    padded_w = w + 2 * pad_width

    padded_img = np.zeros((padded_h, padded_w), dtype=int)
    # Đặt ảnh gốc vào trung tâm
    padded_img[pad_width : pad_width + h, pad_width : pad_width + w] = image

    return padded_img


def loc_trung_vi_3x3(image):
    """
    Tự cài đặt bộ lọc trung vị 3x3.
    """
    h, w = image.shape
    output = np.zeros((h, w), dtype=np.uint8)

    # Bước 1: Pad ảnh (lề 1 pixel cho filter 3x3)
    padded = pad_anh_thu_cong(image, pad_width=1)

    # Bước 2: Duyệt qua từng pixel của ảnh gốc
    for i in range(h):
        for j in range(w):
            # Lấy vùng lân cận 3x3 từ ảnh đã pad
            # Tọa độ trong pad lệch +1 so với ảnh gốc
            region = padded[i : i + 3, j : j + 3]

            # Bước 3: Sắp xếp giá trị (chuyển về mảng 1 chiều rồi sort)
            sorted_region = np.sort(region.flatten())

            # Bước 4: Lấy giá trị trung vị (phần tử thứ 4 trong mảng 9 phần tử)
            median_val = sorted_region[4]

            output[i, j] = median_val

    return output


def main():
    # 1. Đọc ảnh xám
    img_path = "image.jpg"  # Thay đường dẫn ảnh của bạn vào đây
    try:
        img = Image.open(img_path).convert("L")
    except FileNotFoundError:
        print(f"Không tìm thấy file {img_path}")
        return

    img_arr = np.array(img)

    # 4. Thêm nhiễu muối tiêu (5% - 10%)
    noisy_img_arr = them_nhieu_muoi_tieu(img_arr, prob=0.08)  # 8%

    # 3. Lọc trung vị để khử nhiễu
    denoised_img_arr = loc_trung_vi_3x3(noisy_img_arr)

    # 5. Hiển thị kết quả
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
