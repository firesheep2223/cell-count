# CellPose 簡易影像分析系統
**版本 1.0** | 簡化的 CellPose 影像分割與面積計算流程

---

## 🎯 功能概述

本系統提供完整的細胞影像分析解決方案，整合 CellPose 深度學習模型進行自動化細胞分割與定量分析。

### 核心功能
- **智能分割**：採用 CellPose 先進演算法進行精確細胞邊界識別
- **批次處理**：自動批次讀取資料夾內所有影像檔案
- **面積計算**：精確計算每個細胞面積（μm²），支援自動尺度偵測
- **GPU 加速**：可選 GPU 加速運算，提升處理效率
- **視覺化輸出**：生成黑白遮罩與分析視覺化結果

---

## 🔧 系統需求與安裝

### 環境設置
建議在虛擬環境中進行安裝以避免套件衝突：

```bash
# 建立虛擬環境
python -m venv cellpose_env

# 啟用環境
# Windows
cellpose_env\Scripts\activate
# macOS/Linux  
source cellpose_env/bin/activate
```

### 核心套件安裝
```bash
pip install cellpose numpy pillow tifffile opencv-python matplotlib
```

> **⚡ GPU 加速設置**  
> 如需使用 GPU 加速，請先安裝相容的 PyTorch + CUDA，再在程式中啟用 `model = models.Cellpose(gpu=True, ...)`

---

## 📋 快速開始指南

### 基本使用流程

```python
from cellpose import models
import numpy as np

# 步驟 1：讀取影像資料
image_list, filenames = upload_images()

# 步驟 2：初始化 CellPose 模型
model = models.Cellpose(
    gpu=False,          # 設為 True 啟用 GPU
    model_type="cyto3"  # 細胞分割模型
)

# 步驟 3：執行影像分割
masks_pred, flows, styles, diams = model.eval(
    image_list, 
    diameter=10,        # 細胞平均直徑（像素）
    channels=[0,0],     # 通道設定：[細胞質, 細胞核]
    niter=200          # 迭代次數
)

# 步驟 4：計算細胞面積
cell_areas = calculate_cell_areas_from_masks(
    masks_pred, 
    image_list=image_list, 
    pixel_size=0.5,     # μm/pixel，或設為 None 自動偵測
    visualize=False
)
```

### 通道設定說明 (CHANNELS)

CellPose 需要正確設定影像通道以進行最佳分割效果：

**通道定義**：
- `0` = 灰階
- `1` = 紅色通道 (Red)
- `2` = 綠色通道 (Green)  
- `3` = 藍色通道 (Blue)

**channels 參數格式**：`[細胞質通道, 細胞核通道]`

> **💡 重要提醒**：如果沒有細胞核通道，將第二個參數設為 0

#### 常見設定範例

**單一影像類型（所有影像相同）**：
```python
channels = [0,0]   # 灰階影像，無細胞核通道
channels = [2,3]   # 綠色=細胞質，藍色=細胞核
channels = [2,1]   # 綠色=細胞質，紅色=細胞核
channels = [1,0]   # 紅色=細胞質，無細胞核通道
```

**混合影像類型（每張影像不同通道）**：
```python
channels = [[2,3], [0,0], [0,0]]  # 第一張為彩色，其餘為灰階
channels = [[1,3], [2,1], [0,0]]  # 每張影像使用不同通道組合
```

### 參數調整建議

| 參數 | 建議值 | 說明 |
|------|--------|------|
| `diameter` | 10-30 | 根據實際細胞大小調整 |
| `model_type` | `"cyto3"` | 一般細胞；`"nuclei"` 適用於細胞核 |
| `channels` | `[0,0]` | 見上方通道設定說明 |
| `niter` | 200-500 | 增加可提升分割精度但耗時較長 |
| `pixel_size` | 實測值 | 如不確定可設為 `None` 使用自動偵測 |

---

## 📊 輸出結果

### 檔案輸出
- **`output_mask/`**：每張影像對應的黑白遮罩檔案（TIFF 格式）
- **`cell_areas`**：包含每張影像中各細胞面積資料的陣列（單位：μm²）

### 資料結構
```python
# cell_areas 範例輸出
[
    [245.6, 312.8, 189.3],  # 第一張影像的細胞面積
    [298.1, 267.4, 334.2],  # 第二張影像的細胞面積
    # ...
]
```

---

## 🗂️ 專案結構

```
CellPose_Analysis/
├── input_cell/              # 輸入影像資料夾
│   ├── image_001.tif
│   ├── image_002.tif
│   └── ...
├── output_mask/             # 輸出遮罩資料夾
│   ├── mask_001.tif
│   ├── mask_002.tif
│   └── ...
├── src/
│   ├── main.py             # 主程式：影像讀取、分割、遮罩儲存
│   └── utils.py            # 工具函數：尺度偵測、面積計算、視覺化
├── requirements.txt        # 套件需求清單
└── README.md              # 專案說明文件
```

---

## ⚠️ 常見問題排除

### GPU 記憶體不足
**症狀**：出現 CUDA OOM 錯誤  
**解決方案**：
- 改用 CPU 模式：`gpu=False`
- 降低批次大小或影像解析度
- 清理 GPU 記憶體：`torch.cuda.empty_cache()`

### 尺度偵測失敗
**症狀**：無法自動偵測 pixel_size  
**解決方案**：
- 調整 `cv2.threshold` 閾值參數
- 進行影像前處理增強對比度
- 手動測量並輸入已知的 pixel_size 值

### 面積計算異常
**症狀**：計算出的面積與預期差異過大  
**解決方案**：
- 確認 pixel_size 設定正確
- 檢查分割遮罩品質（建議啟用視覺化）
- 驗證影像尺度與實際樣本對應關係

---

## 🎓 學術應用範例

此系統特別適用於生物醫學研究中的細胞形態分析，例如：
- 藥物處理後的細胞面積變化分析
- 不同培養條件下的細胞生長評估
- 螢光標記細胞的自動化計數與測量

---


# 論文資訊頁面

<div align="center">

## 國立臺灣大學理學院化學系
### 碩士論文

**Department of Chemistry**  
**College of Science**  
**National Taiwan University**  
**Master's Thesis**

---

### 藉由光保護基調變優化具光可控性的紫杉醇螢光探針

**Optimize Light-Controlled Restore of Paclitaxel Activity and Fluorescence through Dual-Photocage Tuning**

---

**研究生**：郭雨彤  
**Student**：Yu-Tong Kuo

**學位**：碩士論文  
**Degree**：Master's Thesis

---

*國立臺灣大學化學系*  
*Department of Chemistry, National Taiwan University*

</div>
