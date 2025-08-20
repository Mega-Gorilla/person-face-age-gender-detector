"""
最適化された人物・顔検出器
GPU自動検出、バッチ処理、キャッシング、フレームスキップ機能を統合
"""

import numpy as np
import cv2
import torch
from ultralytics import YOLO
from typing import List, Dict, Optional, Tuple, Any
import logging
import time
from collections import deque
from dataclasses import dataclass, field
import threading
from concurrent.futures import ThreadPoolExecutor

from src.utils.gpu_manager import gpu_manager
from src.core.face_detector import FaceDetector
from src.core.age_gender_caffe import CaffeAgeGenderEstimator

logger = logging.getLogger(__name__)


@dataclass
class DetectionCache:
    """検出結果のキャッシュ"""
    frame_id: int
    timestamp: float
    persons: List[Dict] = field(default_factory=list)
    faces: List[Dict] = field(default_factory=list)
    age_gender: List[Dict] = field(default_factory=list)
    
    def is_valid(self, max_age: float = 0.1) -> bool:
        """キャッシュが有効かチェック（max_age秒以内）"""
        return (time.time() - self.timestamp) < max_age


class OptimizedDetector:
    """最適化された統合検出器"""
    
    def __init__(
        self,
        yolo_model: str = "yolo11n.pt",
        confidence_threshold: float = 0.5,
        face_confidence: float = 0.8,
        enable_face: bool = True,
        enable_age_gender: bool = True,
        use_gpu: bool = True,
        batch_size: Optional[int] = None,
        cache_size: int = 10,
        skip_frames: int = 0
    ):
        """
        最適化検出器の初期化
        
        Args:
            yolo_model: YOLOモデル名
            confidence_threshold: 人物検出の信頼度閾値
            face_confidence: 顔検出の信頼度閾値
            enable_face: 顔検出を有効化
            enable_age_gender: 年齢性別推定を有効化
            use_gpu: GPU使用フラグ
            batch_size: バッチサイズ（Noneで自動設定）
            cache_size: キャッシュサイズ
            skip_frames: スキップするフレーム数
        """
        self.confidence_threshold = confidence_threshold
        self.face_confidence = face_confidence
        self.enable_face = enable_face
        self.enable_age_gender = enable_age_gender
        self.skip_frames = skip_frames
        
        # GPU設定
        self.device = gpu_manager.get_device(prefer_gpu=use_gpu)
        self.torch_device = gpu_manager.get_torch_device(prefer_gpu=use_gpu)
        
        # バッチサイズを自動設定
        if batch_size is None:
            self.batch_size = gpu_manager.get_optimal_batch_size("yolo")
        else:
            self.batch_size = batch_size
        
        logger.info(f"最適化検出器を初期化: デバイス={self.device}, バッチサイズ={self.batch_size}")
        
        # モデル初期化
        self._initialize_models(yolo_model)
        
        # キャッシュ初期化
        self.cache = deque(maxlen=cache_size)
        self.frame_counter = 0
        self.last_detection = None
        
        # バッチ処理用バッファ
        self.frame_buffer = []
        self.buffer_lock = threading.Lock()
        
        # 並列処理用エグゼキュータ
        self.executor = ThreadPoolExecutor(max_workers=2)
        
        # パフォーマンス統計
        self.stats = {
            'total_frames': 0,
            'processed_frames': 0,
            'skipped_frames': 0,
            'cache_hits': 0,
            'avg_inference_time': 0,
            'avg_fps': 0
        }
        self.timing_buffer = deque(maxlen=30)
        
    def _initialize_models(self, yolo_model: str) -> None:
        """モデルの初期化"""
        try:
            # YOLO人物検出モデル
            logger.info(f"YOLOモデル {yolo_model} を読み込み中...")
            self.yolo_model = YOLO(yolo_model)
            
            # GPUへ移動
            if 'cuda' in self.device:
                self.yolo_model.to(self.device)
            
            # 推論モードに設定して最適化
            self.yolo_model.model.eval()
            if 'cuda' in self.device:
                # 最初の推論でJITコンパイル（ウォームアップ）
                dummy_input = torch.zeros(1, 3, 640, 640).to(self.torch_device)
                with torch.no_grad():
                    _ = self.yolo_model.model(dummy_input)
            
            logger.info(f"YOLOモデルを {self.device} に配置完了")
            
            # 顔検出モデル
            if self.enable_face:
                use_gpu_face = 'cuda' in self.device
                self.face_detector = FaceDetector(use_gpu=use_gpu_face)
                logger.info(f"顔検出モデルを初期化 (GPU: {use_gpu_face})")
            else:
                self.face_detector = None
            
            # 年齢性別推定モデル
            if self.enable_age_gender:
                use_gpu_age = 'cuda' in self.device
                self.age_gender_estimator = CaffeAgeGenderEstimator(use_gpu=use_gpu_age)
                logger.info(f"年齢性別推定モデルを初期化 (GPU: {use_gpu_age})")
            else:
                self.age_gender_estimator = None
                
        except Exception as e:
            logger.error(f"モデル初期化エラー: {e}")
            raise
    
    def detect(self, frame: np.ndarray, force_process: bool = False) -> Dict:
        """
        フレームから検出を実行（最適化版）
        
        Args:
            frame: 入力フレーム
            force_process: 強制的に処理を実行
            
        Returns:
            検出結果の辞書
        """
        start_time = time.time()
        self.stats['total_frames'] += 1
        self.frame_counter += 1
        
        # フレームスキップチェック
        if not force_process and self.skip_frames > 0:
            if self.frame_counter % (self.skip_frames + 1) != 0:
                self.stats['skipped_frames'] += 1
                # 前回の結果を返す
                if self.last_detection:
                    self.stats['cache_hits'] += 1
                    return self.last_detection
                else:
                    return self._empty_result()
        
        # キャッシュチェック
        cache_result = self._check_cache(frame)
        if cache_result is not None:
            self.stats['cache_hits'] += 1
            self.last_detection = cache_result
            return cache_result
        
        # 実際の検出処理
        self.stats['processed_frames'] += 1
        
        # バッチ処理の準備
        if self.batch_size > 1:
            result = self._batch_detect(frame)
        else:
            result = self._single_detect(frame)
        
        # キャッシュに保存
        self._update_cache(frame, result)
        self.last_detection = result
        
        # タイミング統計更新
        inference_time = time.time() - start_time
        self.timing_buffer.append(inference_time)
        self.stats['avg_inference_time'] = np.mean(self.timing_buffer)
        self.stats['avg_fps'] = 1.0 / self.stats['avg_inference_time'] if self.stats['avg_inference_time'] > 0 else 0
        
        return result
    
    def _single_detect(self, frame: np.ndarray) -> Dict:
        """単一フレーム検出処理"""
        result = {
            'persons': [],
            'faces': [],
            'frame_id': self.frame_counter,
            'timestamp': time.time()
        }
        
        # 人物検出
        with torch.no_grad():
            yolo_results = self.yolo_model(frame, verbose=False, device=self.device)
        
        persons = self._process_yolo_results(yolo_results)
        result['persons'] = persons
        
        # 顔検出と年齢性別推定を並列実行
        if self.enable_face and persons:
            # 各人物領域で顔検出
            face_futures = []
            for person in persons:
                future = self.executor.submit(
                    self._detect_faces_in_person,
                    frame,
                    person
                )
                face_futures.append((person, future))
            
            # 結果を収集
            for person, future in face_futures:
                faces = future.result(timeout=0.5)
                if faces:
                    person['faces'] = faces
                    result['faces'].extend(faces)
        
        return result
    
    def _batch_detect(self, frame: np.ndarray) -> Dict:
        """バッチ検出処理（複数フレームを同時処理）"""
        # バッファに追加
        with self.buffer_lock:
            self.frame_buffer.append(frame)
            
            # バッファがバッチサイズに達していない場合は単一処理
            if len(self.frame_buffer) < self.batch_size:
                return self._single_detect(frame)
            
            # バッチ処理実行
            batch_frames = self.frame_buffer[:self.batch_size]
            self.frame_buffer = self.frame_buffer[self.batch_size:]
        
        # バッチ推論
        with torch.no_grad():
            batch_results = self.yolo_model(batch_frames, verbose=False, device=self.device)
        
        # 最後のフレームの結果を返す
        result = {
            'persons': [],
            'faces': [],
            'frame_id': self.frame_counter,
            'timestamp': time.time()
        }
        
        if batch_results:
            persons = self._process_yolo_results([batch_results[-1]])
            result['persons'] = persons
            
            # 顔検出
            if self.enable_face and persons:
                for person in persons:
                    faces = self._detect_faces_in_person(frame, person)
                    if faces:
                        person['faces'] = faces
                        result['faces'].extend(faces)
        
        return result
    
    def _detect_faces_in_person(self, frame: np.ndarray, person: Dict) -> List[Dict]:
        """人物領域内で顔検出と年齢性別推定を実行"""
        faces = []
        
        if not self.face_detector:
            return faces
        
        # 人物バウンディングボックスを取得
        x1, y1, x2, y2 = person['bbox']
        
        # 顔検出
        detected_faces = self.face_detector.detect(frame, person_bbox=(x1, y1, x2, y2))
        
        # 年齢性別推定
        if self.enable_age_gender and self.age_gender_estimator and detected_faces:
            for face in detected_faces:
                fx1, fy1, fx2, fy2 = face['bbox']
                face_roi = frame[fy1:fy2, fx1:fx2]
                
                if face_roi.size > 0:
                    try:
                        age, gender = self.age_gender_estimator.estimate(face_roi)
                        face['age'] = age
                        face['gender'] = gender
                        face['age_range'] = self._age_to_range(age) if age else "Unknown"
                    except Exception as e:
                        logger.debug(f"年齢性別推定エラー: {e}")
                        face['age'] = None
                        face['gender'] = "Unknown"
                        face['age_range'] = "Unknown"
        
        faces.extend(detected_faces)
        return faces
    
    def _process_yolo_results(self, results) -> List[Dict]:
        """YOLO結果を処理"""
        person_detections = []
        
        for result in results:
            if result.boxes is None:
                continue
            
            for box in result.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                
                # 人物クラス（ID=0）のみ
                if class_id == 0 and confidence >= self.confidence_threshold:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    detection = {
                        'bbox': (x1, y1, x2, y2),
                        'confidence': confidence,
                        'center': ((x1 + x2) // 2, (y1 + y2) // 2),
                        'area': (x2 - x1) * (y2 - y1),
                        'faces': []
                    }
                    person_detections.append(detection)
        
        return person_detections
    
    def _check_cache(self, frame: np.ndarray) -> Optional[Dict]:
        """キャッシュをチェック"""
        # 簡単なフレーム比較（ハッシュベース）
        # 実際の実装では、より洗練された比較方法を使用
        return None
    
    def _update_cache(self, frame: np.ndarray, result: Dict) -> None:
        """キャッシュを更新"""
        cache_entry = DetectionCache(
            frame_id=self.frame_counter,
            timestamp=time.time(),
            persons=result.get('persons', []),
            faces=result.get('faces', [])
        )
        self.cache.append(cache_entry)
    
    def _empty_result(self) -> Dict:
        """空の結果を返す"""
        return {
            'persons': [],
            'faces': [],
            'frame_id': self.frame_counter,
            'timestamp': time.time()
        }
    
    def _age_to_range(self, age: Optional[int]) -> str:
        """年齢を範囲に変換"""
        if age is None:
            return "Unknown"
        elif age < 18:
            return "0-17"
        elif age < 30:
            return "18-29"
        elif age < 45:
            return "30-44"
        elif age < 60:
            return "45-59"
        else:
            return "60+"
    
    def update_settings(
        self,
        confidence_threshold: Optional[float] = None,
        face_confidence: Optional[float] = None,
        skip_frames: Optional[int] = None,
        enable_face: Optional[bool] = None,
        enable_age_gender: Optional[bool] = None
    ) -> None:
        """設定を動的に更新"""
        if confidence_threshold is not None:
            self.confidence_threshold = confidence_threshold
        if face_confidence is not None:
            self.face_confidence = face_confidence
        if skip_frames is not None:
            self.skip_frames = skip_frames
        if enable_face is not None:
            self.enable_face = enable_face
        if enable_age_gender is not None:
            self.enable_age_gender = enable_age_gender
    
    def get_stats(self) -> Dict:
        """パフォーマンス統計を取得"""
        return {
            **self.stats,
            'device': self.device,
            'batch_size': self.batch_size,
            'skip_frames': self.skip_frames,
            'cache_size': len(self.cache),
            'system_info': gpu_manager.get_system_info()
        }
    
    def cleanup(self) -> None:
        """リソースのクリーンアップ"""
        self.executor.shutdown(wait=False)
        gpu_manager.clear_cache()
        logger.info("最適化検出器のクリーンアップ完了")