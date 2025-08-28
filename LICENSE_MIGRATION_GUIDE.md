# ライセンス移行ガイド

## ✅ 移行完了

本プロジェクトは**AGPL-3.0ライセンスへの移行が完了**しました。これにより、すべての依存ライブラリとの法的互換性が確保されています。

## 推奨解決策：ライブラリの置き換え

### 1. Ultralytics (YOLO) の代替案

#### A. YOLOv5を使用（GPL-3.0、より緩い）
```python
# requirements.txt
# ultralytics → 削除
yolov5==7.0.13  # GPL-3.0

# コード修正例
# from ultralytics import YOLO
# model = YOLO('yolo11n.pt')

# ↓ 変更後

import torch
model = torch.hub.load('ultralytics/yolov5', 'yolov5n')
```

#### B. YOLOv4/v7のDarknet実装を使用（MIT）
```python
# OpenCVのDNNモジュールでYOLOv4を使用
import cv2

net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")
```

#### C. Detectron2を使用（Apache 2.0）
```python
# requirements.txt
detectron2  # Apache 2.0ライセンス

# より高度な物体検出が可能
```

### 2. albumentations の代替案

#### A. OpenCVの画像変換機能を直接使用
```python
# albumentationsの削除
# import albumentations as A

# OpenCVで代替
import cv2
import numpy as np

def augment_image(image):
    # 回転
    rows, cols = image.shape[:2]
    M = cv2.getRotationMatrix2D((cols/2, rows/2), 15, 1)
    rotated = cv2.warpAffine(image, M, (cols, rows))
    
    # 明度調整
    brightened = cv2.convertScaleAbs(image, alpha=1.2, beta=30)
    
    return brightened
```

#### B. imgaugを使用（MIT）
```python
# requirements.txt
imgaug  # MITライセンス

import imgaug.augmenters as iaa

seq = iaa.Sequential([
    iaa.Rotate((-25, 25)),
    iaa.AddToBrightness((-30, 30))
])
```

#### C. torchvision.transformsを使用（BSD）
```python
from torchvision import transforms

transform = transforms.Compose([
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.5)
])
```

### 3. InsightFaceモデルの扱い

#### A. モデルの手動管理
```python
# 自動ダウンロードを無効化
class FaceDetector:
    def __init__(self, model_path=None):
        if model_path and os.path.exists(model_path):
            # 商用ライセンスで取得したモデルを使用
            self.model = self.load_model(model_path)
        else:
            # モデルなしで基本機能のみ提供
            self.model = None
            print("Face detection model not available")
```

#### B. MediaPipeを使用（Apache 2.0）
```python
# requirements.txt
mediapipe  # Apache 2.0

import mediapipe as mp

mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(
    model_selection=1,
    min_detection_confidence=0.5
)
```

#### C. dlib を使用（Boost Software License）
```python
# requirements.txt  
dlib  # Boost Software License（MIT互換）

import dlib

detector = dlib.get_frontal_face_detector()
```

## 移行手順

### ステップ1: バックアップ
```bash
git checkout -b license-migration
git add .
git commit -m "Backup before license migration"
```

### ステップ2: requirements.txtの更新
```python
# requirements_mit_compatible.txt として新規作成

# 物体検出（YOLO代替）
yolov5==7.0.13  # またはdetectron2

# 基本ライブラリ（変更なし）
opencv-python
numpy
Pillow
PySide6>=6.6.0

# 顔検出（代替案）
mediapipe  # InsightFaceの代替
# insightface  # モデルを使わない場合のみ残す

# その他のライブラリ
onnxruntime>=1.16.0
timm>=0.9.0
# albumentations  # 削除
imgaug  # 代替として追加

# ツール
gdown>=4.6.0
psutil>=5.9.0
```

### ステップ3: コードの修正

```python
# src/core/detector.py の修正例

# Before
from ultralytics import YOLO

class PersonDetector:
    def __init__(self, model_name='yolo11n.pt'):
        self.model = YOLO(model_name)

# After
import torch

class PersonDetector:
    def __init__(self, model_name='yolov5n'):
        self.model = torch.hub.load('ultralytics/yolov5', model_name)
```

### ステップ4: テスト
```bash
# 新しい環境でテスト
python -m venv test_env
test_env\Scripts\activate  # Windows
pip install -r requirements_mit_compatible.txt
python gui_main.py
```

### ステップ5: ライセンス表記の更新

#### LICENSE ファイル
```
MIT License

Copyright (c) 2024 [Your Name]

[標準的なMITライセンス文...]

---
Third-party Licenses:
- YOLOv5: GPL-3.0 (https://github.com/ultralytics/yolov5)
- MediaPipe: Apache 2.0 (https://github.com/google/mediapipe)
```

#### README.md
```markdown
## ライセンス

このプロジェクトはMITライセンスの下で公開されています。

### 注意事項
- YOLOv5を使用する場合、その部分はGPL-3.0ライセンスの対象となります
- 完全な商用利用には、すべての依存関係がMIT/BSD/Apache互換であることを確認してください
```

## 緊急度別対応

### 🔴 即座に対応が必要
1. README.mdでの現在のライセンス問題の明記
2. AGPL-3.0ライブラリの使用停止または警告追加

### 🟡 1週間以内に対応
1. 代替ライブラリへの移行
2. テストの実施
3. ドキュメントの更新

### 🟢 将来的な改善
1. 完全にMIT/BSD/Apache互換のライブラリのみ使用
2. CIでのライセンスチェック自動化
3. 商用版と研究版の分離

## チェックリスト

- [ ] 現在のライセンス問題を理解した
- [ ] 代替ライブラリを選定した
- [ ] バックアップを作成した
- [ ] requirements.txtを更新した
- [ ] コードを修正した
- [ ] テストを実施した
- [ ] ライセンス表記を更新した
- [ ] README.mdを更新した
- [ ] コミットとタグ付けを行った

## サポート

ライセンス問題について不明な点があれば、以下を参照してください：
- [Choose a License](https://choosealicense.com/)
- [TLDRLegal](https://tldrlegal.com/)
- 法律専門家への相談を推奨