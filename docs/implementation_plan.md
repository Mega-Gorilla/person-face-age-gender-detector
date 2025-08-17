# 顔検出・年齢性別推定 実装プラン

## 現状分析

### 実装済み機能
- ✅ YOLOv11による人物検出 (`src/core/detector.py`)
- ✅ PySide6 GUI（Stream/Fileタブ）
- ✅ リアルタイム処理・動画処理
- ✅ H.264動画圧縮
- ✅ 統計表示・エクスポート機能

### 現在のアーキテクチャ
```
Camera/Video → YOLOv11(人物検出) → Visualizer → Display
```

## 提案する新アーキテクチャ

### パイプライン設計
```
Camera/Video → YOLOv11(人物検出) → SCRFD(顔検出) → MiVOLO(年齢/性別) → Visualizer → Display
                    ↓                     ↓              ↓
              person_boxes           face_boxes    age/gender
```

## 実装計画

### Phase 1: 基盤整備（推定工数: 2-3時間）

#### 1.1 新規モジュール構造の作成
```python
src/
├── core/
│   ├── detector.py         # 既存: 人物検出
│   ├── face_detector.py    # 新規: SCRFD顔検出
│   └── age_gender.py       # 新規: MiVOLO年齢性別推定
├── models/
│   ├── face/              # 新規: 顔検出モデル格納
│   └── age_gender/        # 新規: 年齢性別モデル格納
└── pipelines/
    └── detection_pipeline.py  # 新規: 統合パイプライン
```

#### 1.2 依存関係の追加
```txt
# requirements.txt に追加
insightface>=0.7.3      # SCRFD用
onnxruntime>=1.16.0      # モデル推論用
timm>=0.9.0             # Vision Transformer用
albumentations>=1.3.0    # 画像前処理用
```

### Phase 2: SCRFD顔検出の実装（推定工数: 3-4時間）

#### 2.1 FaceDetectorクラスの実装
```python
class FaceDetector:
    """SCRFD顔検出クラス"""
    
    def __init__(self, model_name="SCRFD_10G_KPS"):
        # SCRFDモデルの初期化
        pass
    
    def detect_faces(self, frame, person_boxes):
        # 人物領域から顔を検出
        # 高速化: 人物領域のみを処理
        pass
    
    def get_face_landmarks(self, face):
        # 顔のランドマーク取得（オプション）
        pass
```

#### 2.2 統合ポイント
- PersonDetectorの検出結果を受け取り、各人物bbox内で顔検出
- 効率化: 全画面ではなく人物領域のみ処理

### Phase 3: MiVOLO年齢性別推定の実装（推定工数: 4-5時間）

#### 3.1 AgeGenderEstimatorクラスの実装
```python
class AgeGenderEstimator:
    """MiVOLO年齢性別推定クラス"""
    
    def __init__(self, model_path="mivolo_v2.onnx"):
        # MiVOLOモデルの初期化
        pass
    
    def estimate(self, frame, face_box, person_box=None):
        # 年齢と性別を推定
        # MiVOLOの特徴: 顔+体の情報を活用
        return {
            'age': estimated_age,
            'age_range': age_bracket,
            'gender': gender,
            'confidence': confidence
        }
```

#### 3.2 統合ポイント
- 顔検出結果を受け取り、各顔に対して推定
- MiVOLOの特徴である顔+体の両方を活用

### Phase 4: パイプライン統合（推定工数: 3-4時間）

#### 4.1 DetectionPipelineクラスの実装
```python
class DetectionPipeline:
    """統合検出パイプライン"""
    
    def __init__(self):
        self.person_detector = PersonDetector()
        self.face_detector = FaceDetector()
        self.age_gender_estimator = AgeGenderEstimator()
    
    def process_frame(self, frame):
        # 1. 人物検出
        persons = self.person_detector.detect(frame)
        
        # 2. 各人物に対して顔検出
        for person in persons:
            faces = self.face_detector.detect_faces(frame, person['bbox'])
            
            # 3. 各顔に対して年齢性別推定
            for face in faces:
                age_gender = self.age_gender_estimator.estimate(
                    frame, face['bbox'], person['bbox']
                )
                face.update(age_gender)
            
            person['faces'] = faces
        
        return persons
```

### Phase 5: GUI統合（推定工数: 2-3時間）

#### 5.1 Visualizerの拡張
```python
def draw_enhanced_detections(self, frame, detections):
    # 人物bbox描画
    # 顔bbox描画
    # 年齢・性別情報の表示
    # 例: "Male, 25-30" or "Female, 18-22"
```

#### 5.2 コントロールパネルの拡張
- 顔検出ON/OFF
- 年齢性別推定ON/OFF
- 検出モード選択（高速/高精度）

#### 5.3 統計情報の拡張
- 検出人数
- 性別分布
- 年齢分布（ヒストグラム）

### Phase 6: 最適化（推定工数: 2-3時間）

#### 6.1 パフォーマンス最適化
- バッチ処理: 複数の顔を一度に推定
- 非同期処理: 検出と推定を並列化
- キャッシング: 同一人物の追跡時に結果を再利用

#### 6.2 精度向上
- アンサンブル: 複数モデルの結果を統合
- 後処理: 時系列での平滑化

## 実装上の重要な考慮事項

### 1. モジュール性の維持
- 各コンポーネントは独立して動作可能
- 個別にON/OFF可能
- テスト可能な設計

### 2. パフォーマンス目標
- リアルタイム処理: 15-20 FPS維持
- GPU利用時: 30+ FPS
- CPU利用時: 10+ FPS

### 3. エラーハンドリング
- モデルダウンロード失敗時のフォールバック
- 顔検出失敗時の処理継続
- メモリ不足時の自動調整

### 4. プライバシー配慮
- 顔のぼかしオプション
- 年齢を範囲で表示
- データの保存制御

## 技術的詳細

### モデルサイズと要件
- SCRFD_10G: ~17MB
- MiVOLO: ~50MB
- 総メモリ使用量: ~2GB (GPU)

### 推論速度（目安）
- 人物検出: 10ms
- 顔検出: 5ms/person
- 年齢性別: 8ms/face
- 合計: ~25ms/frame (40 FPS理論値)

## 実装優先順位

1. **必須機能**
   - 顔検出基本実装
   - 年齢性別推定基本実装
   - GUI表示

2. **推奨機能**
   - パフォーマンス最適化
   - 統計表示
   - エクスポート機能

3. **オプション機能**
   - 顔認識（個人識別）
   - 感情認識
   - 顔のぼかし

## リスクと対策

| リスク | 影響 | 対策 |
|--------|------|------|
| モデルダウンロード失敗 | 高 | ローカルキャッシュ、複数ミラー |
| パフォーマンス低下 | 中 | 段階的処理、フレームスキップ |
| メモリ不足 | 中 | モデル軽量版の使用 |
| 精度不足 | 低 | アンサンブル、後処理 |

## 成功指標

- ✅ リアルタイム処理（15+ FPS）
- ✅ 年齢推定誤差 < 5歳
- ✅ 性別精度 > 95%
- ✅ 顔検出率 > 90%
- ✅ GUIの応答性維持

## 次のステップ

1. このプランのレビューと承認
2. Phase 1の基盤整備から開始
3. 段階的に機能を追加
4. 各フェーズでテストと検証

## 推定総工数

- 開発: 16-20時間
- テスト: 4-5時間
- ドキュメント: 2時間
- **合計: 22-27時間**