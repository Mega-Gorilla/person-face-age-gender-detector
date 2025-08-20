"""
最適化されたStream処理ワーカー
アダプティブフレームスキップ、動的品質調整、パフォーマンス監視機能を統合
"""

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal, QMutex, QWaitCondition
from PySide6.QtGui import QImage
import logging
import time
from typing import Optional, Dict
from collections import deque
import threading

from src.core.optimized_detector import OptimizedDetector
from src.core.camera import CameraCapture
from src.ui.visualizer import Visualizer
from src.utils.gpu_manager import gpu_manager

logger = logging.getLogger(__name__)


class AdaptivePerformanceController:
    """アダプティブパフォーマンスコントローラー"""
    
    def __init__(self, target_fps: float = 30.0):
        """
        Args:
            target_fps: 目標FPS
        """
        self.target_fps = target_fps
        self.target_frame_time = 1.0 / target_fps
        
        # パフォーマンス統計
        self.frame_times = deque(maxlen=30)
        self.current_fps = 0.0
        self.avg_frame_time = 0.0
        
        # アダプティブ設定
        self.skip_frames = 0
        self.quality_level = 1.0  # 1.0 = 最高品質, 0.5 = 低品質
        self.batch_size = 1
        
        # 調整パラメータ
        self.adjustment_interval = 30  # フレーム数
        self.frame_counter = 0
        
    def update(self, frame_time: float) -> None:
        """フレーム時間を更新して調整"""
        self.frame_times.append(frame_time)
        self.frame_counter += 1
        
        if len(self.frame_times) > 5:
            self.avg_frame_time = np.mean(self.frame_times)
            self.current_fps = 1.0 / self.avg_frame_time if self.avg_frame_time > 0 else 0
            
            # 定期的に設定を調整
            if self.frame_counter % self.adjustment_interval == 0:
                self._adjust_settings()
    
    def _adjust_settings(self) -> None:
        """パフォーマンスに基づいて設定を調整"""
        fps_ratio = self.current_fps / self.target_fps
        
        if fps_ratio < 0.5:
            # 非常に遅い：積極的に最適化
            self.skip_frames = min(self.skip_frames + 2, 5)
            self.quality_level = max(self.quality_level - 0.2, 0.5)
            logger.debug(f"パフォーマンス低下検出: skip_frames={self.skip_frames}, quality={self.quality_level:.1f}")
            
        elif fps_ratio < 0.8:
            # 遅い：軽度の最適化
            self.skip_frames = min(self.skip_frames + 1, 3)
            self.quality_level = max(self.quality_level - 0.1, 0.7)
            
        elif fps_ratio > 1.2:
            # 十分速い：品質を向上
            if self.skip_frames > 0:
                self.skip_frames = max(self.skip_frames - 1, 0)
            else:
                self.quality_level = min(self.quality_level + 0.1, 1.0)
            logger.debug(f"パフォーマンス向上: skip_frames={self.skip_frames}, quality={self.quality_level:.1f}")
    
    def get_scaled_resolution(self, original_width: int, original_height: int) -> tuple:
        """品質レベルに基づいてスケールされた解像度を取得"""
        if self.quality_level >= 0.95:
            return (original_width, original_height)
        
        scale = max(self.quality_level, 0.5)
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)
        
        # 32の倍数に調整（一部のモデルで必要）
        new_width = (new_width // 32) * 32
        new_height = (new_height // 32) * 32
        
        return max(new_width, 320), max(new_height, 240)


class OptimizedStreamWorker(QThread):
    """最適化されたStream処理ワーカー"""
    
    # シグナル定義
    frame_ready = Signal(QImage)
    stats_updated = Signal(dict)
    error_occurred = Signal(str)
    detection_results = Signal(dict)
    performance_info = Signal(dict)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # コンポーネント
        self.detector = None
        self.camera = None
        self.visualizer = Visualizer()
        self.performance_controller = AdaptivePerformanceController(target_fps=30)
        
        # スレッド制御
        self.mutex = QMutex()
        self.wait_condition = QWaitCondition()
        self.is_running = False
        self.is_paused = False
        
        # 設定
        self.model_name = "yolo11n.pt"
        self.confidence_threshold = 0.5
        self.face_confidence = 0.8
        self.camera_index = 0
        self.resolution = (1280, 720)
        self.target_fps = 30
        self.show_center = False
        
        # 機能フラグ
        self.enable_face_detection = False
        self.enable_age_gender = False
        self.enable_optimization = True
        self.enable_gpu = True
        
        # 統計
        self.frame_count = 0
        self.detection_count = 0
        self.last_update_time = time.time()
        
    def initialize_components(self) -> bool:
        """コンポーネントの初期化"""
        try:
            # GPU情報をログ出力
            system_info = gpu_manager.get_system_info()
            logger.info(f"システム情報: GPU={system_info['cuda_available']}, "
                       f"選択デバイス={system_info['selected_device']}")
            
            # カメラ初期化
            self.camera = CameraCapture(
                camera_index=self.camera_index,
                width=self.resolution[0],
                height=self.resolution[1],
                fps=self.target_fps
            )
            
            if not self.camera.is_opened():
                raise RuntimeError("カメラの初期化に失敗しました")
            
            # 最適化検出器の初期化
            self.detector = OptimizedDetector(
                yolo_model=self.model_name,
                confidence_threshold=self.confidence_threshold,
                face_confidence=self.face_confidence,
                enable_face=self.enable_face_detection,
                enable_age_gender=self.enable_age_gender,
                use_gpu=self.enable_gpu,
                skip_frames=0  # 初期値、後で動的に調整
            )
            
            logger.info("最適化ワーカーの初期化完了")
            return True
            
        except Exception as e:
            logger.error(f"コンポーネント初期化エラー: {e}")
            self.error_occurred.emit(str(e))
            return False
    
    def run(self):
        """ワーカースレッドのメイン処理"""
        if not self.initialize_components():
            return
        
        self.is_running = True
        frame_timer = time.time()
        
        try:
            while self.is_running:
                # 一時停止チェック
                self.mutex.lock()
                if self.is_paused:
                    self.wait_condition.wait(self.mutex)
                self.mutex.unlock()
                
                if not self.is_running:
                    break
                
                # フレーム処理
                frame_start = time.time()
                self._process_frame()
                frame_time = time.time() - frame_start
                
                # パフォーマンス調整
                if self.enable_optimization:
                    self.performance_controller.update(frame_time)
                    
                    # 検出器の設定を更新
                    self.detector.update_settings(
                        skip_frames=self.performance_controller.skip_frames
                    )
                
                # 統計更新（1秒ごと）
                if time.time() - self.last_update_time > 1.0:
                    self._update_stats()
                    self.last_update_time = time.time()
                
        except Exception as e:
            logger.error(f"ワーカースレッドエラー: {e}")
            self.error_occurred.emit(str(e))
        finally:
            self._cleanup()
    
    def _process_frame(self):
        """フレーム処理"""
        ret, frame = self.camera.read()
        if not ret:
            return
        
        self.frame_count += 1
        original_shape = frame.shape
        
        # アダプティブ解像度調整
        if self.enable_optimization and self.performance_controller.quality_level < 0.95:
            scaled_res = self.performance_controller.get_scaled_resolution(
                frame.shape[1], frame.shape[0]
            )
            if scaled_res != (frame.shape[1], frame.shape[0]):
                frame_scaled = cv2.resize(frame, scaled_res, interpolation=cv2.INTER_LINEAR)
            else:
                frame_scaled = frame
        else:
            frame_scaled = frame
        
        # 検出実行
        results = self.detector.detect(frame_scaled)
        
        # 元の解像度にスケール調整
        if frame_scaled.shape != original_shape:
            scale_x = original_shape[1] / frame_scaled.shape[1]
            scale_y = original_shape[0] / frame_scaled.shape[0]
            
            for person in results.get('persons', []):
                x1, y1, x2, y2 = person['bbox']
                person['bbox'] = (
                    int(x1 * scale_x),
                    int(y1 * scale_y),
                    int(x2 * scale_x),
                    int(y2 * scale_y)
                )
                person['center'] = (
                    int(person['center'][0] * scale_x),
                    int(person['center'][1] * scale_y)
                )
                
                # 顔座標もスケール
                for face in person.get('faces', []):
                    fx1, fy1, fx2, fy2 = face['bbox']
                    face['bbox'] = (
                        int(fx1 * scale_x),
                        int(fy1 * scale_y),
                        int(fx2 * scale_x),
                        int(fy2 * scale_y)
                    )
        
        # 描画（元の解像度で）
        original_frame = self.camera.get_original_frame()
        if original_frame is None:
            original_frame = frame
        
        annotated = self._draw_results(original_frame, results)
        
        # Qt画像に変換して送信
        qt_image = self._convert_to_qt_image(annotated)
        self.frame_ready.emit(qt_image)
        
        # 詳細結果を送信
        self.detection_results.emit(results)
        
        # 検出数を更新
        self.detection_count = len(results.get('persons', []))
    
    def _draw_results(self, frame: np.ndarray, results: Dict) -> np.ndarray:
        """検出結果を描画"""
        annotated = frame.copy()
        
        # パフォーマンス情報を描画
        if self.enable_optimization:
            info_text = [
                f"FPS: {self.performance_controller.current_fps:.1f}",
                f"Skip: {self.performance_controller.skip_frames}",
                f"Quality: {self.performance_controller.quality_level:.0%}",
                f"Device: {self.detector.device}"
            ]
            
            y_offset = 30
            for text in info_text:
                cv2.putText(annotated, text, (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                y_offset += 25
        
        # 検出結果を描画
        for person in results.get('persons', []):
            # 人物ボックス
            x1, y1, x2, y2 = person['bbox']
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # ラベル
            label = f"Person {person['confidence']:.2f}"
            cv2.putText(annotated, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # 中心点
            if self.show_center:
                cv2.circle(annotated, person['center'], 5, (255, 0, 0), -1)
            
            # 顔検出結果
            for face in person.get('faces', []):
                fx1, fy1, fx2, fy2 = face['bbox']
                cv2.rectangle(annotated, (fx1, fy1), (fx2, fy2), (255, 0, 0), 2)
                
                # 年齢性別
                if 'age' in face and 'gender' in face:
                    ag_label = f"{face['gender']}, {face['age']}"
                    cv2.putText(annotated, ag_label, (fx1, fy2 + 15),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        
        return annotated
    
    def _convert_to_qt_image(self, frame: np.ndarray) -> QImage:
        """OpenCV画像をQt画像に変換"""
        height, width, channel = frame.shape
        bytes_per_line = 3 * width
        
        # BGRからRGBに変換
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # QImageを作成
        qt_image = QImage(
            frame_rgb.data,
            width,
            height,
            bytes_per_line,
            QImage.Format_RGB888
        )
        
        return qt_image.copy()
    
    def _update_stats(self):
        """統計情報を更新"""
        detector_stats = self.detector.get_stats()
        
        stats = {
            'fps': self.performance_controller.current_fps,
            'frame_count': self.frame_count,
            'person_count': self.detection_count,
            'total_detections': detector_stats.get('processed_frames', 0),
            'cache_hits': detector_stats.get('cache_hits', 0),
            'skipped_frames': detector_stats.get('skipped_frames', 0),
            'processing_time': self.performance_controller.avg_frame_time,
            'device': detector_stats.get('device', 'cpu'),
            'skip_frames': self.performance_controller.skip_frames,
            'quality_level': self.performance_controller.quality_level
        }
        
        self.stats_updated.emit(stats)
        
        # パフォーマンス情報も送信
        self.performance_info.emit({
            'gpu_info': detector_stats.get('system_info', {}),
            'optimization_enabled': self.enable_optimization,
            'adaptive_settings': {
                'skip_frames': self.performance_controller.skip_frames,
                'quality': self.performance_controller.quality_level,
                'target_fps': self.performance_controller.target_fps
            }
        })
    
    def pause(self):
        """一時停止"""
        self.mutex.lock()
        self.is_paused = True
        self.mutex.unlock()
    
    def resume(self):
        """再開"""
        self.mutex.lock()
        self.is_paused = False
        self.wait_condition.wakeAll()
        self.mutex.unlock()
    
    def stop(self):
        """停止"""
        self.is_running = False
        self.resume()  # 一時停止中の場合は起こす
        self.wait()  # スレッド終了を待つ
    
    def _cleanup(self):
        """クリーンアップ"""
        if self.camera:
            self.camera.release()
        
        if self.detector:
            self.detector.cleanup()
        
        # GPUキャッシュをクリア
        gpu_manager.clear_cache()
        
        logger.info("最適化ワーカーのクリーンアップ完了")
    
    def update_settings(self, settings: Dict):
        """設定を更新"""
        if 'confidence_threshold' in settings:
            self.confidence_threshold = settings['confidence_threshold']
            if self.detector:
                self.detector.update_settings(
                    confidence_threshold=self.confidence_threshold
                )
        
        if 'enable_face_detection' in settings:
            self.enable_face_detection = settings['enable_face_detection']
            if self.detector:
                self.detector.update_settings(
                    enable_face=self.enable_face_detection
                )
        
        if 'enable_age_gender' in settings:
            self.enable_age_gender = settings['enable_age_gender']
            if self.detector:
                self.detector.update_settings(
                    enable_age_gender=self.enable_age_gender
                )
        
        if 'enable_optimization' in settings:
            self.enable_optimization = settings['enable_optimization']
        
        if 'target_fps' in settings:
            self.target_fps = settings['target_fps']
            self.performance_controller.target_fps = self.target_fps