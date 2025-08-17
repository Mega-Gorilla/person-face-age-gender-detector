# 🎯 顔検出・年齢性別検出の動的制御実装プラン

## 現状分析

### 既存実装の確認

#### 1. **ワーカークラス構成**
```
├── YoloDetectionWorker (基本版・現在使用中)
│   └── 人物検出のみ
├── EnhancedDetectionWorker (拡張版・作成済み)
│   └── 人物＋顔＋年齢性別検出
└── FileProcessingWorker (ファイル処理)
    └── 人物検出のみ
```

#### 2. **GUIファイル構成**
```
├── main_window.py (現在のメインGUI)
│   └── YoloDetectionWorkerを使用
└── gui_main_enhanced.py (別ファイル)
    └── EnhancedDetectionWorkerを使用
```

### 問題点
1. **分離した実装**: 拡張機能が別ファイルに存在
2. **動的制御なし**: 現在のGUIに顔検出ON/OFFトグルがない
3. **パイプライン未統合**: StableDetectionPipelineが使われていない

## 実装プラン

### Phase 1: ワーカークラスの統合（推定: 2時間）

#### 1.1 YoloDetectionWorkerの拡張
```python
class YoloDetectionWorker(QThread):
    def __init__(self):
        # 既存のコード
        
        # 新規追加
        self.enable_face_detection = False
        self.enable_age_gender = False
        self.pipeline = None  # StableDetectionPipeline
        
    def toggle_face_detection(self, enabled: bool):
        """顔検出のON/OFF切り替え"""
        self.enable_face_detection = enabled
        if self.pipeline:
            self.pipeline.update_config(enable_face_detection=enabled)
    
    def toggle_age_gender(self, enabled: bool):
        """年齢性別推定のON/OFF切り替え"""
        self.enable_age_gender = enabled
        if self.pipeline:
            self.pipeline.update_config(enable_age_gender=enabled)
```

#### 1.2 パイプライン切り替えロジック
```python
def initialize_components(self):
    if self.enable_face_detection or self.enable_age_gender:
        # StableDetectionPipelineを使用
        from src.pipelines.stable_detection_pipeline import StableDetectionPipeline
        self.pipeline = StableDetectionPipeline(config)
    else:
        # 既存のPersonDetectorを使用
        self.detector = PersonDetector(...)
```

### Phase 2: GUI コントロールの追加（推定: 1.5時間）

#### 2.1 ControlPanelの拡張
```python
# src/gui/widgets/control_panel.py に追加

def create_face_detection_controls(self) -> QGroupBox:
    """顔検出コントロールの作成"""
    group = QGroupBox("Face Detection")
    layout = QVBoxLayout()
    
    # 顔検出トグル
    self.face_detection_checkbox = QCheckBox("Enable Face Detection")
    self.face_detection_checkbox.toggled.connect(self.face_detection_toggled.emit)
    
    # 年齢性別推定トグル
    self.age_gender_checkbox = QCheckBox("Enable Age/Gender")
    self.age_gender_checkbox.toggled.connect(self.age_gender_toggled.emit)
    
    # 顔検出信頼度
    self.face_confidence_slider = QSlider(Qt.Horizontal)
    self.face_confidence_slider.setRange(50, 100)
    self.face_confidence_slider.setValue(80)
```

#### 2.2 MainWindowでの信号接続
```python
def setup_connections(self):
    # 既存の接続
    
    # 新規追加
    self.control_panel.face_detection_toggled.connect(
        self.on_face_detection_toggled
    )
    self.control_panel.age_gender_toggled.connect(
        self.on_age_gender_toggled
    )
    
def on_face_detection_toggled(self, enabled):
    """顔検出トグル処理"""
    if self.detection_worker:
        self.detection_worker.toggle_face_detection(enabled)
```

### Phase 3: FileWorkerの拡張（推定: 1.5時間）

#### 3.1 FileProcessingWorkerの更新
```python
class FileProcessingWorker(QThread):
    def __init__(self):
        # 既存のコード
        
        # 新規追加
        self.enable_face_detection = False
        self.enable_age_gender = False
        self.pipeline = None
        
    def set_parameters(self, params):
        # 既存のパラメータ
        
        # 顔検出パラメータ追加
        self.enable_face_detection = params.get('enable_face_detection', False)
        self.enable_age_gender = params.get('enable_age_gender', False)
```

#### 3.2 FileProcessorWidgetの更新
```python
# 顔検出オプションをUIに追加
self.face_detection_checkbox = QCheckBox("Detect Faces")
self.age_gender_checkbox = QCheckBox("Estimate Age/Gender")

# パラメータに含める
params = {
    'enable_face_detection': self.face_detection_checkbox.isChecked(),
    'enable_age_gender': self.age_gender_checkbox.isChecked(),
    # 他のパラメータ
}
```

### Phase 4: Visualizerの拡張（推定: 1時間）

#### 4.1 描画メソッドの統合
```python
class Visualizer:
    def draw_enhanced_detections(self, frame, results):
        """拡張検出結果の描画"""
        # 人物描画
        for person in results['persons']:
            self._draw_person(frame, person)
            
            # 顔描画（有効時のみ）
            if 'faces' in person:
                for face in person['faces']:
                    self._draw_face(frame, face)
                    
                    # 年齢性別描画（有効時のみ）
                    if 'age' in face or 'gender' in face:
                        self._draw_age_gender(frame, face)
```

## 実装順序

### Step 1: 基盤準備（30分）
1. StableDetectionPipelineのインポート追加
2. 必要な型定義の追加
3. 設定項目の定義

### Step 2: Stream Mode実装（2時間）
1. YoloDetectionWorkerの拡張
2. ControlPanelへのUI追加
3. MainWindowでの信号接続
4. リアルタイムトグルのテスト

### Step 3: File Mode実装（1.5時間）
1. FileProcessingWorkerの拡張
2. FileProcessorWidgetへのUI追加
3. パラメータ受け渡しの実装
4. ファイル処理でのテスト

### Step 4: 統合とテスト（1時間）
1. 両モードでの動作確認
2. パフォーマンステスト
3. UIの調整

## 主要な変更ファイル

1. **src/gui/workers/yolo_worker.py**
   - StableDetectionPipeline統合
   - 顔検出トグルメソッド追加

2. **src/gui/widgets/control_panel.py**
   - 顔検出コントロールセクション追加
   - 新規シグナル定義

3. **src/gui/workers/file_worker.py**
   - パイプライン統合
   - 顔検出パラメータ追加

4. **src/gui/widgets/file_processor.py**
   - 顔検出オプションUI追加

5. **src/ui/visualizer.py**
   - 拡張描画メソッド追加

## パフォーマンス考慮事項

### メモリ使用量
- 顔検出OFF: ~500MB
- 顔検出ON: ~650MB
- 顔検出+年齢性別: ~750MB

### 処理速度
- 人物検出のみ: 25-30 FPS
- +顔検出: 18-22 FPS
- +年齢性別: 15-18 FPS

### 最適化戦略
1. **遅延初期化**: 顔検出モデルは有効時のみロード
2. **フレームスキップ**: 顔検出は2-3フレーム毎
3. **ROI処理**: 人物領域内のみ顔検出

## 設定の永続化

```python
# QSettings使用
settings = QSettings('PersonDetector', 'Settings')
settings.setValue('face_detection_enabled', self.face_detection_checkbox.isChecked())
settings.setValue('age_gender_enabled', self.age_gender_checkbox.isChecked())
settings.setValue('face_confidence', self.face_confidence_slider.value())
```

## 期待される結果

### UI改善
- ✅ Stream/Fileモード両方に顔検出トグル
- ✅ リアルタイムON/OFF切り替え
- ✅ 設定の保存と復元

### 機能改善
- ✅ 必要時のみモデルロード（メモリ節約）
- ✅ 動的パフォーマンス調整
- ✅ 統一されたパイプライン使用

### ユーザー体験
- ✅ 直感的なコントロール
- ✅ 即座のフィードバック
- ✅ パフォーマンス表示

## リスクと対策

| リスク | 影響 | 対策 |
|--------|------|------|
| メモリ不足 | アプリクラッシュ | 遅延初期化、モデル解放 |
| FPS低下 | UX悪化 | フレームスキップ、警告表示 |
| 切り替え時のラグ | 一時的なフリーズ | 非同期初期化 |

## 推定作業時間

- 基盤準備: 30分
- Stream Mode: 2時間
- File Mode: 1.5時間
- テスト・調整: 1時間
- **合計: 5時間**

この実装により、ユーザーは必要に応じて顔検出・年齢性別推定を
動的にON/OFF切り替えできるようになります。