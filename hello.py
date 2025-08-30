import os
import numpy as np
import matplotlib.pyplot as plt
from cellpose import models, utils, io, plot, transforms
from PIL import Image
import io as sysio
from tifffile import imsave
from cellpose import io

io.logger_setup() # 執行此行以顯示進度

# 上傳並處理圖片
def upload_images():
    """
    上傳圖片並轉換為 NumPy 陣列格式的 RGB 影像。

    回傳：
        image_list (list): 包含轉換後影像的清單，每個影像形狀為 (H, W, C)。
    """
    dir = ".\\input_cell\\"
    filenames = os.listdir(dir)
    print(filenames)
    image_list = []
    for file_name in filenames:
        file_name = dir + file_name
        img = Image.open(file_name).convert("RGB")  # 讀取影像並轉換為 RGB
        img_array = np.array(img)  # 轉換為 NumPy 陣列 (H, W, C)
        image_list.append(img_array)
    return image_list, filenames

def predict_img(imgs, masks_pred):
    """
    繪製原始影像及其對應的預測遮罩輪廓。

    參數：
        imgs (list): 原始影像清單，每個影像形狀為 (H, W, C)。
        masks_pred (list): 預測遮罩清單。
    """
    plt.figure(figsize=(12, 6))  # 設定繪圖區域大小
    titles = [str(i) for i in range(len(imgs))]

    for iex in range(len(imgs)):
        img = imgs[iex].copy()  # 複製原始影像
        ax = plt.subplot(3, 8, (iex % 3) * 8 + (iex // 3) + 1)  # 設定子圖位置
        ax.imshow(img)  # 顯示原始影像

        # 設定軸範圍
        ax.set_ylim([0, img.shape[0]])
        ax.set_xlim([0, img.shape[1]])

        # 繪製預測遮罩的輪廓
        outlines_pred = utils.outlines_list(masks_pred[iex])
        for o in outlines_pred:
            plt.plot(o[:, 0], o[:, 1], color=[1, 1, 0.3], lw=0.75, ls="--")  # 黃色虛線

        # 關閉軸顯示
        plt.axis('off')

        # 設定每組影像的標題
        if iex % 1 == 0:
            ax.set_title(titles[iex // 1])

    # 調整子圖間距並顯示
    plt.tight_layout()
    plt.show()

# 上傳圖片並執行範例
image_list, filenames = upload_images()

# predict_img(image_list, masks_pred)  # 在此替換為你的預測遮罩



# 定義 CELLPOSE 模型及尺寸
# model_type='cyto3' 或 model_type='nuclei'
model = models.Cellpose(gpu=True, model_type="cyto3")

# 定義進行分割的通道 (CHANNELS)
# 灰階=0, 紅=1, 綠=2, 藍=3
# channels = [細胞質, 細胞核]
# 如果沒有細胞核通道，將第二個通道設為 0
# channels = [0,0]
# 如果所有影像類型相同，可用一個包含兩個元素的列表
# channels = [0,0] # 如果影像為灰階
# channels = [2,3] # 如果 G=細胞質，B=細胞核
# channels = [2,1] # 如果 G=細胞質，R=細胞核
# 或者，如果每張影像的通道類型不同
# channels = [[2,3], [0,0], [0,0]]

# 如果使用自訂模型或其他內建模型：
# model = models.CellposeModel(gpu=True, model_type="livecell_cp3")



import numpy as np
import matplotlib.pyplot as plt
import cv2

# 面積閾值，用於過濾細胞面積
AREA_THRESHOLD = 2

def calculate_pixel_size(image: np.ndarray, line_length_microns: float = 50, visualize: bool = False) -> float:
    """
    根據影像中的白色長線計算每像素的實際大小（微米）。

    參數:
    - image: numpy.ndarray
        RGB 影像。
    - line_length_microns: float
        長線的實際長度（微米）。
    - visualize: bool, optional
        是否視覺化長線檢測結果。

    回傳:
    - pixel_size: float
        每像素的大小（微米）。
    """
    # 轉換為灰度並進行二值化
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    # 找輪廓，兼容 OpenCV 3.x 和 4.x
    contours_info = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours_info[1] if len(contours_info) == 3 else contours_info[0]

    print(f"找到 {len(contours)} 個輪廓")

    # 過濾有效輪廓
    valid_contours = [cnt for cnt in contours if cnt is not None and len(cnt) >= 3]
    if not valid_contours:
        raise ValueError("未檢測到有效輪廓，請檢查影像或調整閾值設定。")

    # 找到最大輪廓
    max_contour = max(valid_contours, key=cv2.contourArea)

    # 計算長線的像素長度
    x, y, w, h = cv2.boundingRect(max_contour)
    line_length_pixels = max(w, h)

    # 計算每像素的大小
    pixel_size = line_length_microns / line_length_pixels

    # 視覺化
    if visualize:
        image_copy = image.copy()
        # 將顏色從白色改為紅色，表示偵測到尺標
        cv2.rectangle(image_copy, (x, y), (x + w, y + h), (255, 0, 0), 2)
        plt.imshow(image_copy)
        plt.title(f"Detected Line: {line_length_pixels} pixels, {pixel_size:.3f} μm/pixel")
        plt.axis('off')
        plt.show()

    return pixel_size


def calculate_cell_areas_from_masks(
    masks: np.ndarray ,
    image_list: list = None,
    pixel_size: float = None,
    visualize: bool = False
) -> dict:
    """
    根據分割結果計算細胞面積（單位：平方微米）。

    參數:
    - masks: numpy.ndarray 或 list
        分割結果的標籤影像或多影像列表。
    - image_list: list, optional
        原始影像列表，與 masks 對應，用於多影像情況下計算像素大小。
    - pixel_size: float, optional
        每像素的實際大小（微米）。
    - visualize: bool, optional
        是否視覺化結果。

    回傳:
    - cell_areas_micron: dict
        每個影像中，每個細胞的面積（單位：平方微米）。
    """
    if isinstance(masks, list):
        if image_list is None:
            raise ValueError("當 masks 為列表時，必須提供對應的 image_list。")
        all_cell_areas = {}
        for i, mask in enumerate(masks):
            current_pixel_size = pixel_size
            if current_pixel_size is None:
                current_pixel_size = calculate_pixel_size(image_list[i], line_length_microns=50, visualize=False)
            # visualize_cell_contours(image_list[i], mask,current_pixel_size)
            visualize_cell_contours_black_and_white(image_list[i], mask,current_pixel_size)
            filename = filenames[i].split(".")[0]
            generate_black_white_mask(mask,f"output_mask\\{filename}.tiff",current_pixel_size)
            cell_areas = calculate_cell_areas_from_masks(mask, pixel_size=current_pixel_size, visualize=visualize)
            print(f"影像 {i} 的像素大小: {current_pixel_size} μm/pixel")
            all_cell_areas[f'image_{i}'] = cell_areas
        return all_cell_areas

    # 單一影像處理
    unique_cells = np.unique(masks)
    cell_areas = {cell: np.sum(masks == cell) for cell in unique_cells if cell > 0}
    cell_areas_micron = {cell: area * (pixel_size ** 2) for cell, area in cell_areas.items()}

    if visualize:
        plt.imshow(masks, cmap='jet')
        for cell, area in cell_areas_micron.items():
            if area > AREA_THRESHOLD:
                y, x = np.mean(np.argwhere(masks == cell), axis=0)
                plt.text(x, y, f'{area:.1f} μm²', color='white', fontsize=10, ha='center')
        plt.title('Cell Areas (μm²)')
        plt.axis('off')
        plt.show()

    return cell_areas_micron


def visualize_cell_contours(image: np.ndarray, masks: np.ndarray, pixel_size: float, area_threshold: float = AREA_THRESHOLD) -> dict:
    """
    在原始影像上為每個細胞繪製黃色虛線邊框，並填充半透明顏色。

    參數:
    - image: numpy.ndarray
        原始影像。
    - masks: numpy.ndarray
        分割結果的標籤影像。
    - pixel_size: float
        每像素的實際大小（微米）。
    - area_threshold: float, optional
        細胞面積的閾值，低於此值的細胞將被忽略。

    回傳:
    - cell_areas_micron: dict
        每個細胞的面積（單位：平方微米）。
    """
    image_copy = image.copy()
    overlay = image.copy()

    # 獲取所有細胞的輪廓
    outlines_pred = utils.outlines_list(masks)

    cell_areas_micron = {}
    unique_cells = np.unique(masks)
    unique_cells = unique_cells[unique_cells > 0]  # 排除背景

    for cell in unique_cells:
        cell_mask = (masks == cell).astype(np.uint8)
        # 獲取當前細胞的輪廓
        cell_outlines = utils.outlines_list(cell_mask)
        if not cell_outlines:
            continue  # 如果沒有輪廓，跳過

        # 計算細胞面積
        area_pixels = np.sum(cell_mask)
        area_micron = area_pixels * (pixel_size ** 2)
        if area_micron < area_threshold:
            continue  # 忽略小於閾值的細胞

        cell_areas_micron[cell] = area_micron

        for outline in cell_outlines:
            # 繪製填充顏色
            cv2.drawContours(overlay, [outline.astype(np.int32)], -1, (255, 255, 0), -1)  # 填充黃色

            # 繪製黃色虛線邊框
            plt.plot(outline[:, 0], outline[:, 1], color=[1, 1, 0.3], lw=0.75, ls="--")  # 黃色虛線

    # 調整透明度
    alpha = 0.4
    cv2.addWeighted(overlay, alpha, image_copy, 1 - alpha, 0, image_copy)

    plt.imshow(image_copy)
    plt.title('Cell Contours with Yellow Dashed Lines')
    plt.axis('off')
    plt.show()

    return cell_areas_micron



def visualize_cell_contours_black_and_white(image: np.ndarray, masks: np.ndarray, pixel_size: float, area_threshold: float = AREA_THRESHOLD) -> dict:
    """
    在原始影像上為每個細胞繪製黃色虛線邊框，並填充半透明顏色。

    參數:
    - image: numpy.ndarray
        原始影像。
    - masks: numpy.ndarray
        分割結果的標籤影像。
    - pixel_size: float
        每像素的實際大小（微米）。
    - area_threshold: float, optional
        細胞面積的閾值，低於此值的細胞將被忽略。

    回傳:
    - cell_areas_micron: dict
        每個細胞的面積（單位：平方微米）。
    """
    image_copy = image.copy()
    overlay = image.copy()

    # 獲取所有細胞的輪廓
    outlines_pred = utils.outlines_list(masks)

    cell_areas_micron = {}
    unique_cells = np.unique(masks)
    unique_cells = unique_cells[unique_cells > 0]  # 排除背景

    for cell in unique_cells:
        cell_mask = (masks == cell).astype(np.uint8)
        # 獲取當前細胞的輪廓
        cell_outlines = utils.outlines_list(cell_mask)
        if not cell_outlines:
            continue  # 如果沒有輪廓，跳過

        # 計算細胞面積
        area_pixels = np.sum(cell_mask)
        area_micron = area_pixels * (pixel_size ** 2)
        if area_micron < area_threshold:
            continue  # 忽略小於閾值的細胞

        cell_areas_micron[cell] = area_micron

        for outline in cell_outlines:
            # 繪製填充顏色
            cv2.drawContours(overlay, [outline.astype(np.int32)], -1, (255, 255, 255), 3)  # 填充黑色

            # 繪製黃色虛線邊框
            plt.plot(outline[:, 0], outline[:, 1], color=[0, 0, 0], lw=1, ls="-")  # 白色實線

    # 調整透明度
    alpha = 1
    cv2.addWeighted(overlay, alpha, image_copy, 1 - alpha, 0, image_copy)

    plt.imshow(image_copy)
    plt.title('Cell Contours with Black White')
    plt.axis('off')
    plt.show()

    return cell_areas_micron


def generate_black_white_mask(masks: np.ndarray, output_path: str, pixel_size):
    """
    生成黑色背景、白色填充細胞內部輪廓的 TIFF mask。

    參數:
    - masks: numpy.ndarray
        分割結果的標籤影像。
    - output_path: str
        保存生成 TIFF mask 的路徑。
    """
    # 初始化全黑背景
    black_white_mask = np.zeros_like(masks, dtype=np.uint8)

    unique_cells = np.unique(masks)
    unique_cells = unique_cells[unique_cells > 0]  # 排除背景

    for cell in unique_cells:
        cell_mask = (masks == cell).astype(np.uint8)
        area_pixels = np.sum(cell_mask)
        area_micron = area_pixels * (pixel_size ** 2)
        print(area_micron)
        if area_micron < AREA_THRESHOLD:
            continue  # 忽略小於閾值的細胞


        # 填充細胞內部輪廓為白色
        black_white_mask[cell_mask > 0] = 255
        # 提取細胞的輪廓
        cell_outlines = utils.outlines_list(cell_mask)

        # 將輪廓部分設置為黑色
        for outline in cell_outlines:
            outline_x = outline[:, 0].astype(int)  # 確保轉換為整數
            outline_y = outline[:, 1].astype(int)  # 確保轉換為整數
            black_white_mask[outline_y,outline_x] = 0


    # 保存為 TIFF 格式
    imsave(output_path, black_white_mask)



masks_pred, flows, styles, diams = model.eval(image_list, diameter=200, channels=[2,0],
                                              niter=2000) # 對細菌影像使用更多迭代


# predict_img(image_list, masks_pred)  # 在此替換為你的預測遮罩

cell_areas = calculate_cell_areas_from_masks(masks_pred,image_list, pixel_size=50, visualize=False)
