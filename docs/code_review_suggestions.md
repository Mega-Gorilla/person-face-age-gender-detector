# コードレビューと改善提案

## 現在のコードの強み 💪

1. **モジュール性**: 各コンポーネントが適切に分離されている
2. **エラーハンドリング**: 適切な例外処理とロギング
3. **GUI設計**: タブ分離、ワーカースレッド使用
4. **パフォーマンス**: FPS監視、統計表示
5. **拡張性**: 新機能追加が容易な構造

## 改善提案と統合ポイント 🎯

### 1. src/core/detector.py の拡張案

**現状**: PersonDetectorクラスのみ

**提案**: 基底クラスを作成して継承構造に

```python
# src/core/base_detector.py
class BaseDetector(ABC):
    """検出器の基底クラス"""
    @abstractmethod
    def detect(self, frame: np.ndarray) -> List[Dict]:
        pass

# src/core/detector.py
class PersonDetector(BaseDetector):
    """既存の人物検出クラス"""
    # 現在の実装を維持

# src/core/face_detector.py  
class FaceDetector(BaseDetector):
    """新規: SCRFD顔検出クラス"""
    def detect(self, frame: np.ndarray, roi: Optional[Tuple] = None) -> List[Dict]:
        # ROI指定で人物領域内のみ検出可能
        pass
```

### 2. パイプライン統合の提案

**新規ファイル**: `src/pipelines/detection_pipeline.py`

```python
class EnhancedDetectionPipeline:
    """階層的検出パイプライン"""
    
    def __init__(self, config: Dict):
        self.person_detector = PersonDetector()
        self.face_detector = None  # 遅延初期化
        self.age_gender_estimator = None  # 遅延初期化
        self.enable_face = config.get('enable_face', True)
        self.enable_age_gender = config.get('enable_age_gender', True)
    
    def process_frame(self, frame: np.ndarray) -> Dict:
        """
        階層的処理:
        1. 人物検出（必須）
        2. 顔検出（オプション）
        3. 年齢性別推定（オプション）
        """
        result = {
            'persons': [],
            'faces': [],
            'stats': {}
        }
        
        # 人物検出
        persons = self.person_detector.detect(frame)
        
        # 顔検出（有効時のみ）
        if self.enable_face and persons:
            self._detect_faces(frame, persons)
        
        # 年齢性別推定（有効時のみ）
        if self.enable_age_gender:
            self._estimate_age_gender(frame, persons)
        
        return self._format_results(persons)
```

### 3. GUI統合の改善案

**src/gui/widgets/control_panel.py の拡張**

```python
# 新規セクション追加
class FaceDetectionSettings(QGroupBox):
    """顔検出設定パネル"""
    def __init__(self):
        super().__init__("Face Detection")
        self.setup_ui()
    
    def setup_ui(self):
        # 顔検出ON/OFF
        self.enable_face = QCheckBox("Enable Face Detection")
        # 年齢性別推定ON/OFF  
        self.enable_age_gender = QCheckBox("Enable Age/Gender")
        # モデル選択
        self.model_combo = QComboBox()
        self.model_combo.addItems(["Fast", "Balanced", "Accurate"])
```

### 4. Visualizerの拡張案

**src/ui/visualizer.py の改善**

```python
class EnhancedVisualizer(Visualizer):
    """拡張ビジュアライザー"""
    
    def draw_person_with_face_info(self, frame, person_data):
        """人物と顔情報を統合表示"""
        
        # 人物bbox（緑）
        self.draw_bbox(frame, person_data['bbox'], color=(0, 255, 0))
        
        # 顔bbox（青）
        for face in person_data.get('faces', []):
            self.draw_bbox(frame, face['bbox'], color=(255, 0, 0))
            
            # 年齢性別表示
            if 'age' in face and 'gender' in face:
                label = f"{face['gender']}, {face['age_range']}"
                self.draw_label(frame, face['bbox'], label)
    
    def draw_statistics_overlay(self, frame, stats):
        """統計オーバーレイ"""
        # 人数、性別分布、年齢分布を表示
        pass
```

### 5. パフォーマンス最適化の提案

**バッチ処理の実装**

```python
class BatchProcessor:
    """バッチ処理による高速化"""
    
    def process_batch_faces(self, faces: List[np.ndarray]) -> List[Dict]:
        """複数の顔を一度に処理"""
        # ONNXRuntimeのバッチ推論を活用
        batch_input = np.stack(faces)
        results = self.model.run(batch_input)
        return self._parse_batch_results(results)
```

**非同期処理の活用**

```python
from concurrent.futures import ThreadPoolExecutor

class AsyncPipeline:
    """非同期パイプライン"""
    
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=2)
    
    def process_frame_async(self, frame):
        # 顔検出と年齢性別推定を並列実行
        future_faces = self.executor.submit(self.detect_faces, frame)
        future_age = self.executor.submit(self.estimate_age, frame)
        return future_faces, future_age
```

### 6. エラーハンドリングの強化

```python
class ModelManager:
    """モデル管理クラス"""
    
    def load_model_with_fallback(self, model_name: str):
        """フォールバック付きモデル読み込み"""
        try:
            return self.load_primary_model(model_name)
        except ModelNotFoundError:
            logger.warning(f"Primary model {model_name} not found")
            return self.load_fallback_model()
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return self.load_minimal_model()
```

### 7. テスト容易性の向上

```python
# src/core/mock_detector.py
class MockFaceDetector:
    """テスト用モック検出器"""
    def detect(self, frame):
        # テスト用の固定結果を返す
        return [
            {'bbox': (100, 100, 200, 200), 
             'confidence': 0.95,
             'age': 25,
             'gender': 'male'}
        ]
```

## 段階的実装アプローチ 📝

### Step 1: 基盤準備（影響小）
- 基底クラスの作成
- 設定ファイルの拡張
- 依存関係の追加

### Step 2: 顔検出実装（中規模変更）
- FaceDetectorクラス実装
- PersonDetectorとの統合
- 基本的なGUI表示

### Step 3: 年齢性別推定（機能追加）
- AgeGenderEstimatorクラス実装
- パイプライン統合
- 結果の可視化

### Step 4: 最適化（パフォーマンス向上）
- バッチ処理実装
- 非同期処理導入
- キャッシング機構

## 互換性の維持 🔄

既存機能への影響を最小限に：

1. **後方互換性**: 既存のAPIは変更しない
2. **オプトイン**: 新機能はデフォルトOFF
3. **段階的移行**: 古い実装と新実装の共存期間
4. **設定による切り替え**: config.jsonで新旧切り替え

## メモリとパフォーマンスの考慮 💾

### メモリ使用量の見積もり
- 現在: ~500MB (YOLOv11n)
- 追加: ~70MB (SCRFD + MiVOLO)
- 合計: ~570MB

### FPS目標
- CPU: 10-15 FPS
- GPU: 25-30 FPS
- 軽量モード: 20+ FPS

## 推奨される実装順序 🎯

1. **Week 1**: 基盤整備とSCRFD統合
2. **Week 2**: MiVOLO統合とGUI改善
3. **Week 3**: 最適化とテスト

## まとめ

現在のコードは良好な構造を持っており、新機能の統合は比較的スムーズに行えます。提案した改善により：

- ✅ 最小限の既存コード変更
- ✅ モジュール性の維持
- ✅ 段階的な機能追加
- ✅ パフォーマンスの最適化
- ✅ テスト容易性の向上

これらの改善により、顔検出・年齢性別推定機能を効率的に統合できます。