# 🚀 即実装可能なアクションリスト

## 今すぐ始められる実装ステップ

### ⚡ Phase 1: 基盤準備（30分で完了可能）

#### 1. 依存関係の追加
```bash
# requirements.txt に追加
pip install insightface>=0.7.3
pip install onnxruntime>=1.16.0
pip install timm>=0.9.0
```

#### 2. ディレクトリ構造の作成
```bash
mkdir -p src/pipelines
mkdir -p src/models/face
mkdir -p src/models/age_gender
touch src/core/face_detector.py
touch src/core/age_gender.py
touch src/pipelines/detection_pipeline.py
```

### ⚡ Phase 2: 最小実装（2時間で動作確認）

#### 1. シンプルな顔検出クラス
```python
# src/core/face_detector.py
import insightface
from insightface.app import FaceAnalysis

class FaceDetector:
    def __init__(self):
        self.app = FaceAnalysis(providers=['CPUExecutionProvider'])
        self.app.prepare(ctx_id=0, det_size=(640, 640))
    
    def detect(self, frame, person_bbox=None):
        if person_bbox:
            x1, y1, x2, y2 = person_bbox
            roi = frame[y1:y2, x1:x2]
            faces = self.app.get(roi)
            # 座標を元画像に変換
            for face in faces:
                face.bbox[:2] += [x1, y1]
                face.bbox[2:] += [x1, y1]
        else:
            faces = self.app.get(frame)
        return faces
```

#### 2. 既存Detectorへの統合
```python
# src/core/detector.py に追加
def detect_with_faces(self, frame):
    """人物検出と顔検出を統合"""
    persons = self.detect(frame)
    
    if hasattr(self, 'face_detector'):
        for person in persons:
            faces = self.face_detector.detect(frame, person['bbox'])
            person['faces'] = faces
    
    return persons
```

### ⚡ Phase 3: GUI即座対応（1時間）

#### 1. Visualizerの簡易拡張
```python
# src/ui/visualizer.py に追加
def draw_face_info(self, frame, person):
    """顔情報を追加描画"""
    # 既存の人物bbox描画
    self.draw_detections(frame, [person])
    
    # 顔bbox追加（青色）
    for face in person.get('faces', []):
        x1, y1, x2, y2 = face['bbox']
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
        # 簡易的な年齢性別表示
        if 'age' in face:
            label = f"Age: {face['age']}"
            cv2.putText(frame, label, (x1, y1-10), 
                       self.font, 0.5, (255, 255, 255), 1)
```

### ⚡ Phase 4: テスト実装（30分）

#### テストスクリプト
```python
# debug/test_face_detection.py
import sys
sys.path.append('..')

from src.core.detector import PersonDetector
from src.core.face_detector import FaceDetector
import cv2

def test_pipeline():
    # 初期化
    person_det = PersonDetector()
    face_det = FaceDetector()
    
    # テスト画像
    img = cv2.imread('debug/sample.jpg')
    
    # 検出実行
    persons = person_det.detect(img)
    for person in persons:
        faces = face_det.detect(img, person['bbox'])
        print(f"Person found, {len(faces)} faces detected")
    
    print("✅ Test passed!")

if __name__ == "__main__":
    test_pipeline()
```

## 📋 優先順位付きTODOリスト

### 🔴 必須（今日中）
- [ ] insightfaceインストール
- [ ] face_detector.py作成
- [ ] 基本的な顔検出実装
- [ ] 既存GUIで顔bbox表示

### 🟡 推奨（今週中）
- [ ] MiVOLOモデルダウンロード
- [ ] 年齢性別推定実装
- [ ] GUIコントロール追加
- [ ] パフォーマンステスト

### 🟢 オプション（来週以降）
- [ ] バッチ処理最適化
- [ ] 非同期処理実装
- [ ] 詳細な統計表示
- [ ] エクスポート機能拡張

## 🎯 最速実装パス

**目標: 3時間で動作するプロトタイプ**

1. **Hour 1**: 
   - insightfaceインストール（10分）
   - face_detector.py実装（30分）
   - detector.py統合（20分）

2. **Hour 2**:
   - visualizer.py拡張（30分）
   - GUIテスト（30分）

3. **Hour 3**:
   - デバッグ・調整（30分）
   - パフォーマンス確認（30分）

## 💡 実装のコツ

1. **段階的実装**: まず顔検出だけ、次に年齢性別
2. **既存コード活用**: PersonDetectorの構造を参考に
3. **エラー処理後回し**: まず動くものを作る
4. **GUI最小変更**: 既存のdraw_detectionsを拡張

## ⚠️ 注意事項

- **モデルサイズ**: 初回は自動ダウンロード（~100MB）
- **速度**: CPUでも10FPS程度は出る
- **メモリ**: +200MB程度増加
- **互換性**: Python 3.8以上推奨

## 🔧 トラブルシューティング

```bash
# insightfaceインストールエラー時
pip install --upgrade pip
pip install insightface --no-deps
pip install -r requirements_insightface.txt

# ONNXRuntimeエラー時
pip install onnxruntime-cpu  # CPU版
# または
pip install onnxruntime-gpu  # GPU版

# Import エラー時
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

## ✅ 成功の確認方法

1. `python debug/test_face_detection.py` が動く
2. GUIで顔のbboxが青色で表示される
3. FPSが10以上を維持
4. メモリ使用量が1GB以下

---

**Next Action**: 
```bash
pip install insightface onnxruntime
```

これで実装を開始できます！ 🚀