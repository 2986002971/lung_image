import os

import cv2
import numpy as np

scale_factor = 0.1  # 缩放比例


# 读取文件夹中的所有图片
def load_images_from_folder(folder):
    images = []
    # 遍历所有文件
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        if os.path.isfile(img_path) and img_path.endswith(".jpg"):
            img = cv2.imread(img_path)
            if img is not None:
                # 保存文件名和图像
                images.append((filename, img_path, img))
    return images


# 鼠标回调函数，用于选择四个点
def select_points(event, x, y, flags, param):
    global points
    img = param
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x / scale_factor, y / scale_factor))
        cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow("Select Points", img)


# 边缘裁切和矫正
def process_image(img):
    global points
    points = []

    img_resized = cv2.resize(
        img, (int(img.shape[1] * scale_factor), int(img.shape[0] * scale_factor))
    )

    cv2.imshow("Select Points", img_resized)
    cv2.setMouseCallback("Select Points", select_points, param=img_resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if len(points) == 4:
        # 排序点，确保顺序为：左上，右上，右下，左下
        points = np.array(points, dtype="float32")
        rect = np.zeros((4, 2), dtype="float32")

        s = points.sum(axis=1)
        rect[0] = points[np.argmin(s)]
        rect[2] = points[np.argmax(s)]

        diff = np.diff(points, axis=1)
        rect[1] = points[np.argmin(diff)]
        rect[3] = points[np.argmax(diff)]

        (tl, tr, br, bl) = rect

        # 计算新图像的宽度和高度
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA / 2), int(widthB / 2))

        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA / 2), int(heightB / 2))

        # 透视变换
        dst = np.array(
            [
                [0, 0],
                [maxWidth - 1, 0],
                [maxWidth - 1, maxHeight - 1],
                [0, maxHeight - 1],
            ],
            dtype="float32",
        )

        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))

        return warped
    else:
        return img


# 主函数
def main():
    folder_path = "./CT图片"  # 替换为您的输入文件夹路径
    save_folder = "./mycoplasma"  # 替换为您的输出文件夹路径

    # 确保输出文件夹存在
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    images = load_images_from_folder(folder_path)

    for subdir_name, img_path, img in images:
        processed_img = process_image(img)

        # 显示处理后的图像
        print(f"处理文件夹 {subdir_name} 中的图像")
        cv2.imshow("Processed Image", processed_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # 保存处理后的图像，使用子文件夹名称命名
        save_path = os.path.join(save_folder, f"{subdir_name}.jpg")
        cv2.imwrite(save_path, processed_img)
        print(f"已保存至 {save_path}")


if __name__ == "__main__":
    main()
