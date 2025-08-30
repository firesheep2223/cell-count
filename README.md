# CellPose 簡易影像分析 — v1.0

簡化的 CellPose 影像分割與面積計算流程



## 🚀 功能重點（簡短）

* 使用 *CellPose* 進行影像分割（GPU 可選）。
* 自動批次讀取資料夾影像（`upload_images()`）。
* 從分割結果計算每個細胞面積（μm²）；可用已知 pixel\_size 或自動偵測。
* 輸出黑白 mask（TIFF）與簡單視覺化。

---



## 📦 安裝（最小）

\[!NOTE] 建議在虛擬環境中安裝


# 建立並啟用虛擬環境（示意）
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate

pip install cellpose numpy pillow tifffile opencv-python matplotlib

\[!IMPORTANT] 如果要用 GPU，請先安裝相容的 PyTorch + CUDA，再啟用 `model = models.Cellpose(gpu=True, ...)`。


---

## 🔧 最簡使用範例

# 1) 讀取影像（資料夾: ./input_cell/）
image_list, filenames = upload_images()

# 2) 初始化 CellPose（示範：cyto3）
model = models.Cellpose(gpu=False, model_type="cyto3")

# 3) 執行分割（回傳 masks_pred）
masks_pred, flows, styles, diams = model.eval(image_list, diameter=10, channels=[0,0], niter=200)

# 4) 計算面積（若已知 pixel_size，傳入；否則可改為自動偵測）
cell_areas = calculate_cell_areas_from_masks(masks_pred, image_list=image_list, pixel_size=0.5, visualize=False)

\[!TIP] 若影像內含尺度尺（scale bar），可讓 `calculate_pixel_size()` 自動計算 pixel\_size。


---

## 📝 輸出

* `output_mask/`：黑白 TIFF mask（每張影像一檔）
* `cell_areas`：每張影像每個細胞的面積（單位：μm²）

---

## 常見問題（簡短）

\[!WARNING] 若啟用 GPU 遇到 OOM，改用 `gpu=False` 或降低 batch / 圖片尺寸。


*Q：沒偵測到尺度尺？*
A：調整 cv2.threshold 的閾值或先做影像前處理（對比度/形態學處理）。

*Q：為何面積與預期不同？*
A：確認 pixel\_size 是否正確，以及 masks 是否正確分割（建議視覺化檢查）。

---

## 簡短檔案建議

.
├─ input_cell/
├─ output_mask/
├─ src/
│  ├─ main.py       # 包含 upload_images, run cellpose, 儲存 mask
│  └─ utils.py      # pixel_size、計算面積、視覺化
├─ requirements.txt
└─ README.md

---

## 指定文字（論文 / 身份資訊）

國立臺灣大學理學院化學系
碩士論文
Department of Chemistry
College of Science
National Taiwan University
Master’s Thesis
藉由光保護基調變優化具光可控性的紫杉醇螢光探針
Optimize Light-Controlled Restore of Paclitaxel Activity
and Fluorescence through Dual-Photocage Tuning
郭雨彤
Yu-Tong Kuo

碩士論文
