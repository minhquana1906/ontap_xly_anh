import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def tinh_histogram(image_array):
    """
    Tính histogram của ảnh xám.
    Trả về: tần suất (counts) và các bin edges.
    """
    hist, bins = np.histogram(image_array.flatten(), bins=256, range=(0, 256))
    return hist


def phan_vung_nguong(image_array, T):
    """
    Phân vùng ảnh theo công thức:
    g(x,y) = 255 nếu f(x,y) >= T
           = 0   nếu f(x,y) < T
    """
    # Tạo ma trận kết quả toàn số 0 (màu đen)
    output_img = np.zeros_like(image_array)

    # Gán giá trị 255 (trắng) cho các pixel thỏa mãn điều kiện >= T
    # Đây là kỹ thuật Boolean Indexing của Numpy (rất nhanh)
    output_img[image_array >= T] = 255

    return output_img


def main():
    # 1. Đọc ảnh xám
    img_path = "image.jpg"  # Thay đường dẫn ảnh của bạn vào đây
    try:
        img = Image.open(img_path).convert("L")
    except FileNotFoundError:
        print(f"Không tìm thấy file {img_path}")
        return

    img_arr = np.array(img)

    # 2. Tính histogram
    hist = tinh_histogram(img_arr)

    # 3. Chọn ngưỡng T
    # LƯU Ý CHO SINH VIÊN:
    # Hãy chạy code lần đầu để xem biểu đồ Histogram.
    # Sau đó chọn giá trị T tại vị trí "thung lũng" (giữa 2 đỉnh) và cập nhật lại biến này.
    T = 128  # Giá trị mặc định, hãy thay đổi sau khi quan sát

    # 4. Thực hiện phân vùng
    segmented_img = phan_vung_nguong(img_arr, T)

    # 5. Hiển thị kết quả
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Ảnh gốc
    axes[0].imshow(img_arr, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("Ảnh Gốc (Grayscale)")
    axes[0].axis("off")

    # Biểu đồ Histogram
    axes[1].plot(hist, color="black")
    axes[1].axvline(T, color="red", linestyle="--", label=f"Ngưỡng T={T}")
    axes[1].set_title("Histogram & Ngưỡng đã chọn")
    axes[1].set_xlabel("Mức xám")
    axes[1].set_ylabel("Số lượng pixel")
    axes[1].legend()
    axes[1].set_xlim([0, 255])

    # Ảnh sau phân vùng
    axes[2].imshow(segmented_img, cmap="gray", vmin=0, vmax=255)
    axes[2].set_title(f"Ảnh Phân Vùng (Binary)\nT = {T}")
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
