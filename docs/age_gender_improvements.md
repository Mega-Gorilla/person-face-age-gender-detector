# 年齢・性別推定機能 改善提案

## 優先度：高 - すぐに実装可能な改善

### 1. バッチ推論の最適化

```python
# src/core/age_gender_caffe.py に追加

def batch_estimate_optimized(self, face_images: List[np.ndarray]) -> List[Dict]:
    """最適化されたバッチ推論"""
    if not face_images:
        return []
    
    if self.method != "caffe" or not self.age_net or not self.gender_net:
        # フォールバック
        return [self._estimate_fallback(img) for img in face_images]
    
    results = []
    
    # すべての画像をblobに変換（ベクトル化）
    blobs = []
    for face_img in face_images:
        blob = cv2.dnn.blobFromImage(
            face_img,
            scalefactor=1.0,
            size=(227, 227),
            mean=self.MODEL_MEAN_VALUES,
            swapRB=False,
            crop=False
        )
        blobs.append(blob)
    
    # バッチ処理（可能な限りまとめて処理）
    batch_blob = np.vstack(blobs) if len(blobs) > 1 else blobs[0]
    
    # Gender予測
    self.gender_net.setInput(batch_blob)
    gender_preds_batch = self.gender_net.forward()
    
    # Age予測
    self.age_net.setInput(batch_blob)
    age_preds_batch = self.age_net.forward()
    
    # 結果の解析
    for i in range(len(face_images)):
        gender_preds = gender_preds_batch[i] if len(face_images) > 1 else gender_preds_batch[0]
        age_preds = age_preds_batch[i] if len(face_images) > 1 else age_preds_batch[0]
        
        # 通常の処理
        gender_idx = gender_preds.argmax()
        age_idx = age_preds.argmax()
        
        results.append({
            'age': self._get_age_from_range(self.AGE_LIST[age_idx]),
            'age_range': self.AGE_LIST[age_idx],
            'age_confidence': float(age_preds[age_idx]),
            'gender': self.GENDER_LIST[gender_idx],
            'gender_confidence': float(gender_preds[gender_idx]),
            'method': 'caffe_batch'
        })
    
    return results
```

### 2. キャッシング機能の追加

```python
# src/core/age_gender_caffe.py の __init__ に追加

def __init__(self, use_gpu: bool = False, cache_size: int = 100):
    """
    Initialize with caching support
    
    Args:
        use_gpu: Whether to use GPU for inference
        cache_size: Maximum cache entries (0 to disable)
    """
    self.use_gpu = use_gpu
    self.age_net = None
    self.gender_net = None
    self.cache_size = cache_size
    self.cache = {}  # hash -> result
    self.cache_order = []  # LRU tracking
    
    self._initialize_models()

def estimate_with_cache(
    self,
    face_image: np.ndarray,
    track_id: Optional[str] = None
) -> Dict:
    """キャッシュ付き推定"""
    
    # キャッシュキーの生成
    if track_id and self.cache_size > 0:
        cache_key = track_id
    else:
        # 画像ハッシュをキーとして使用
        cache_key = hashlib.md5(face_image.tobytes()).hexdigest()
    
    # キャッシュチェック
    if cache_key in self.cache:
        # LRU更新
        self.cache_order.remove(cache_key)
        self.cache_order.append(cache_key)
        return self.cache[cache_key].copy()
    
    # 通常の推定
    result = self.estimate(face_image)
    
    # キャッシュに保存
    if self.cache_size > 0:
        self.cache[cache_key] = result.copy()
        self.cache_order.append(cache_key)
        
        # キャッシュサイズ制限
        while len(self.cache) > self.cache_size:
            oldest = self.cache_order.pop(0)
            del self.cache[oldest]
    
    return result
```

### 3. 年齢推定の確定的な改善

```python
# src/core/age_gender_caffe.py

def _get_age_from_range(self, age_range: str, confidence: float = 1.0) -> int:
    """確定的な年齢推定（ランダム性なし）"""
    
    # 基本マッピング
    age_mapping = {
        '(0-2)': 1,
        '(4-6)': 5,
        '(8-12)': 10,
        '(15-20)': 18,
        '(25-32)': 28,
        '(38-43)': 40,
        '(48-53)': 50,
        '(60-100)': 70
    }
    
    base_age = age_mapping.get(age_range, 30)
    
    # 信頼度に基づく調整（確定的）
    if confidence < 0.5:
        # 低信頼度：範囲の中央値に近づける
        if age_range == '(0-2)':
            base_age = 1
        elif age_range == '(4-6)':
            base_age = 5
        elif age_range == '(8-12)':
            base_age = 10
        elif age_range == '(15-20)':
            base_age = 17
        elif age_range == '(25-32)':
            base_age = 28
        elif age_range == '(38-43)':
            base_age = 40
        elif age_range == '(48-53)':
            base_age = 50
        else:  # (60-100)
            base_age = 65
    
    return base_age
```

## 優先度：中 - パフォーマンス向上

### 4. 時間的スムージング強化

```python
# src/pipelines/stable_detection_pipeline.py に追加

class TemporalSmoother:
    """時間的スムージング用クラス"""
    
    def __init__(self, window_size: int = 5):
        self.window_size = window_size
        self.history = {}  # track_id -> deque of estimates
    
    def smooth(self, track_id: str, new_estimate: Dict) -> Dict:
        """推定値をスムージング"""
        
        if track_id not in self.history:
            self.history[track_id] = deque(maxlen=self.window_size)
        
        # 履歴に追加
        self.history[track_id].append(new_estimate)
        
        # 平均化
        smoothed = new_estimate.copy()
        
        if len(self.history[track_id]) > 1:
            # 年齢の平均
            ages = [e['age'] for e in self.history[track_id] if e['age'] is not None]
            if ages:
                smoothed['age'] = int(np.mean(ages))
            
            # 性別の多数決
            genders = [e['gender'] for e in self.history[track_id] if e['gender'] not in ['Unknown', 'Model Not Available']]
            if genders:
                from collections import Counter
                gender_counts = Counter(genders)
                smoothed['gender'] = gender_counts.most_common(1)[0][0]
            
            # 信頼度の平均
            age_confs = [e['age_confidence'] for e in self.history[track_id]]
            gender_confs = [e['gender_confidence'] for e in self.history[track_id]]
            
            smoothed['age_confidence'] = np.mean(age_confs)
            smoothed['gender_confidence'] = np.mean(gender_confs)
        
        return smoothed
```

### 5. 詳細なデバッグログ

```python
# src/core/age_gender_caffe.py

import time

def _estimate_caffe_with_debug(self, face_image: np.ndarray) -> Dict:
    """デバッグ情報付き推定"""
    
    start_time = time.time()
    
    # 入力検証
    logger.debug(f"Input shape: {face_image.shape}, dtype: {face_image.dtype}")
    
    try:
        # Blob作成時間計測
        blob_start = time.time()
        blob = cv2.dnn.blobFromImage(
            face_image,
            scalefactor=1.0,
            size=(227, 227),
            mean=self.MODEL_MEAN_VALUES,
            swapRB=False,
            crop=False
        )
        blob_time = time.time() - blob_start
        logger.debug(f"Blob creation: {blob_time*1000:.2f}ms")
        
        # Gender予測時間
        gender_start = time.time()
        self.gender_net.setInput(blob)
        gender_preds = self.gender_net.forward()
        gender_time = time.time() - gender_start
        logger.debug(f"Gender inference: {gender_time*1000:.2f}ms")
        logger.debug(f"Gender predictions: {gender_preds[0]}")
        
        # Age予測時間
        age_start = time.time()
        self.age_net.setInput(blob)
        age_preds = self.age_net.forward()
        age_time = time.time() - age_start
        logger.debug(f"Age inference: {age_time*1000:.2f}ms")
        logger.debug(f"Age predictions: {age_preds[0]}")
        
        # 結果処理
        gender_idx = gender_preds[0].argmax()
        age_idx = age_preds[0].argmax()
        
        total_time = time.time() - start_time
        logger.debug(f"Total estimation time: {total_time*1000:.2f}ms")
        
        return {
            'age': self._get_age_from_range(self.AGE_LIST[age_idx]),
            'age_range': self.AGE_LIST[age_idx],
            'age_confidence': float(age_preds[0][age_idx]),
            'gender': self.GENDER_LIST[gender_idx],
            'gender_confidence': float(gender_preds[0][gender_idx]),
            'method': 'caffe',
            'processing_time_ms': total_time * 1000
        }
        
    except Exception as e:
        logger.error(f"Caffe estimation failed: {e}", exc_info=True)
        return self._estimate_fallback(face_image)
```

## 優先度：低 - 将来的な拡張

### 6. マルチモデルアンサンブル

```python
# src/core/age_gender_ensemble.py (新規ファイル)

class EnsembleAgeGenderEstimator:
    """複数モデルのアンサンブル推定"""
    
    def __init__(self):
        self.estimators = []
        
        # Caffeモデル
        try:
            from src.core.age_gender_caffe import CaffeAgeGenderEstimator
            self.estimators.append(('caffe', CaffeAgeGenderEstimator(), 0.6))
        except:
            pass
        
        # 他のモデル（将来追加）
        # self.estimators.append(('mivolo', MiVOLOEstimator(), 0.8))
        # self.estimators.append(('fairface', FairFaceEstimator(), 0.7))
    
    def estimate(self, face_image: np.ndarray) -> Dict:
        """重み付きアンサンブル推定"""
        
        if not self.estimators:
            return {
                'age': None,
                'age_range': 'No models available',
                'gender': 'No models available',
                'method': 'none'
            }
        
        results = []
        weights = []
        
        for name, estimator, weight in self.estimators:
            try:
                result = estimator.estimate(face_image)
                if result['method'] != 'fallback':
                    results.append(result)
                    weights.append(weight)
            except:
                continue
        
        if not results:
            return self.estimators[0][1]._estimate_fallback(face_image)
        
        # 重み付き平均
        total_weight = sum(weights)
        weights = [w/total_weight for w in weights]
        
        # 年齢の重み付き平均
        age = sum(r['age'] * w for r, w in zip(results, weights) if r['age'])
        
        # 性別の重み付き投票
        male_score = sum(w for r, w in zip(results, weights) if r['gender'] == 'Male')
        female_score = sum(w for r, w in zip(results, weights) if r['gender'] == 'Female')
        gender = 'Male' if male_score > female_score else 'Female'
        
        return {
            'age': int(age),
            'age_range': results[0]['age_range'],  # 最も重みの高いモデルから
            'gender': gender,
            'gender_confidence': max(male_score, female_score),
            'method': 'ensemble'
        }
```

## 実装優先順位まとめ

| 優先度 | 改善項目 | 期待効果 | 実装工数 |
|--------|----------|----------|----------|
| 高 | バッチ推論 | 性能30%向上 | 2時間 |
| 高 | キャッシング | 重複処理削減 | 1時間 |
| 高 | 確定的年齢推定 | 再現性確保 | 30分 |
| 中 | 時間的スムージング | 安定性向上 | 2時間 |
| 中 | 詳細ログ | デバッグ改善 | 1時間 |
| 低 | アンサンブル | 精度向上 | 4時間 |

これらの改善により、パフォーマンスと安定性が大幅に向上します。