import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def tinh_histogram(image_array):
    """Tính histogram sử dụng numpy cho ngắn gọn"""
    # np.histogram trả về (tần suất, các mốc bin)
    hist, _ = np.histogram(image_array.flatten(), bins=256, range=(0, 256))
    return hist


def phan_vung_anh(image_array, t1, t2):
    """
    Phân ảnh thành 3 vùng dựa trên T1 và T2:
    - Vùng tối (< T1) -> 0
    - Vùng trung bình (T1 <= pixel < T2) -> 128
    - Vùng sáng (>= T2) -> 255
    """
    # Tạo ảnh kết quả giống kích thước ảnh gốc
    output_img = np.zeros_like(image_array)

    # Cách 1: Dùng boolean indexing (trực quan)
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
    # 1. Đọc ảnh xám
    img_path = "image.jpg"  # Thay đường dẫn ảnh của bạn
    try:
        img = Image.open(img_path).convert("L")
    except FileNotFoundError:
        print(f"Không tìm thấy file {img_path}")
        return

    img_arr = np.array(img)

    # 2. Tính histogram
    hist = tinh_histogram(img_arr)

    # 3. Chọn ngưỡng T1 và T2
    # Bạn có thể chọn thủ công hoặc dùng công thức.
    # Ở đây chọn mốc 1/3 và 2/3 dải màu (0-255) cho đơn giản.
    T1 = 85  # Ngưỡng tối
    T2 = 170  # Ngưỡng sáng

    # 4. Gán nhãn/Phân vùng
    segmented_img = phan_vung_anh(img_arr, T1, T2)

    # 5. Hiển thị kết quả
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Ảnh gốc
    axes[0].imshow(img_arr, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("Ảnh Gốc")
    axes[0].axis("off")

    # Histogram và ngưỡng
    axes[1].plot(hist, color="black")
    axes[1].axvline(T1, color="red", linestyle="--", label=f"T1={T1}")
    axes[1].axvline(T2, color="blue", linestyle="--", label=f"T2={T2}")
    axes[1].set_title("Histogram & Ngưỡng phân vùng")
    axes[1].legend()
    axes[1].set_xlim([0, 255])

    # Ảnh sau phân vùng
    axes[2].imshow(segmented_img, cmap="gray", vmin=0, vmax=255)
    axes[2].set_title("Ảnh Phân Vùng (3 Mức)")
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
