# YOLOv11 人物検出最適化ガイド

## 現在の実装の評価結果

### ✅ 優れている点

1. **モデル選択**: YOLOv11nは開発・テスト用途に最適
2. **コード構造**: モジュール化された設計で拡張性が高い
3. **パフォーマンス監視**: リアルタイムFPS表示機能実装済み
4. **ユーザビリティ**: インタラクティブな操作が可能

### ⚠️ 改善可能な点

1. **検出精度の向上余地**
2. **群衆シーンでの性能**
3. **トラッキング機能の欠如**

## 推奨最適化

### 1. シーン適応型モデル切り替え

```python
# src/core/adaptive_detector.py
class AdaptivePersonDetector:
    def __init__(self):
        self.models = {
            'fast': YOLO('yolo11n.pt'),      # 高速処理用
            'balanced': YOLO('yolo11s.pt'),   # バランス型
            'accurate': YOLO('yolo11m.pt')    # 高精度
        }
        self.current_mode = 'balanced'
    
    def detect_with_adaptation(self, frame):
        # フレームの複雑度を分析
        complexity = self.analyze_frame_complexity(frame)
        
        # 適切なモデルを選択
        if complexity < 0.3:
            self.current_mode = 'fast'
        elif complexity < 0.7:
            self.current_mode = 'balanced'
        else:
            self.current_mode = 'accurate'
        
        return self.models[self.current_mode](frame)
```

### 2. 人物検出特化の前処理

```python
# src/utils/preprocessing.py
def enhance_for_person_detection(frame):
    # コントラスト強調（人物のエッジを強調）
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.equalizeHist(l)
    enhanced = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    
    # ノイズ除去（小さな誤検出を減らす）
    enhanced = cv2.bilateralFilter(enhanced, 9, 75, 75)
    
    return enhanced
```

### 3. 信頼度の動的調整

```python
# src/core/dynamic_threshold.py
class DynamicThresholdManager:
    def __init__(self, base_threshold=0.5):
        self.base_threshold = base_threshold
        self.history = deque(maxlen=30)
    
    def update_threshold(self, detection_count, frame_brightness):
        # 検出数と明るさに基づいて閾値を調整
        if detection_count > 10:  # 混雑時
            threshold = min(0.7, self.base_threshold + 0.2)
        elif frame_brightness < 50:  # 暗い環境
            threshold = max(0.3, self.base_threshold - 0.2)
        else:
            threshold = self.base_threshold
        
        return threshold
```

### 4. トラッキング機能の追加

```python
# src/core/tracker.py
from ultralytics import YOLO
import supervision as sv

class PersonTracker:
    def __init__(self, model_path='yolo11n.pt'):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()
        
    def detect_and_track(self, frame):
        # 検出
        results = self.model(frame)
        
        # トラッキング
        detections = sv.Detections.from_ultralytics(results[0])
        tracked = self.tracker.update_with_detections(detections)
        
        return tracked
```

### 5. パフォーマンス最適化

```python
# 最適化設定
optimization_config = {
    # バッチ処理
    'batch_size': 1,  # GPUメモリに応じて調整
    
    # 推論最適化
    'half_precision': True,  # FP16使用
    'dynamic': True,  # 動的入力サイズ
    
    # キャッシュ
    'persist': True,  # モデルをメモリに保持
    
    # マルチスレッド
    'workers': 4,  # データローダーのワーカー数
}

# 適用例
model = YOLO('yolo11n.pt')
model.predict(
    source=frame,
    half=optimization_config['half_precision'],
    device='cpu',  # or 'cuda' if available
    persist=optimization_config['persist']
)
```

## 実装優先順位

| 優先度 | 最適化項目 | 期待効果 | 実装難易度 |
|--------|------------|----------|------------|
| 高 | 動的閾値調整 | 精度10%向上 | 低 |
| 高 | 前処理強化 | 暗所性能改善 | 低 |
| 中 | トラッキング | ID追跡可能 | 中 |
| 中 | モデル切り替え | 適応的性能 | 中 |
| 低 | バッチ処理 | 速度向上 | 高 |

## まとめ

現在の実装は基本要件を満たしており、YOLOv11の選択は適切です。上記の最適化を段階的に実装することで、より高性能な人物検出システムへと発展させることができます。