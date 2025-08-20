"""
統合型YOLOv11検出処理ワーカースレッド（顔検出・年齢性別推定対応）
"""

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal, QMutex, QWaitCondition
from PySide6.QtGui import QImage
import logging
from typing import Optional, Dict
import time

from src.core.detector import PersonDetector
from src.core.camera import CameraCapture
from src.ui.visualizer import Visualizer
from src.utils.performance import PerformanceMonitor
from src.pipelines.stable_detection_pipeline import StableDetectionPipeline

logger = logging.getLogger(__name__)


class IntegratedVisualizerEx(Visualizer):
    """拡張ビジュアライザー（顔・年齢性別表示対応）"""
    
    def draw_integrated_detections(
        self,
        frame: np.ndarray,
        results: Dict,
        show_center: bool = False,
        show_confidence: bool = True,
        show_faces: bool = True,
        show_age_gender: bool = True
    ) -> np.ndarray:
        """統合検出結果の描画"""
        annotated = frame.copy()
        
        # パイプライン結果の場合
        if 'persons' in results:
            for person in results['persons']:
                # 人物バウンディングボックス描画
                x1, y1, x2, y2 = person['bbox']
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # 人物ラベル
                if show_confidence:
                    label = f"Person {person['confidence']:.2f}"
                else:
                    label = "Person"
                
                cv2.putText(annotated, label, (x1, y1 - 10),
                           self.font, self.font_scale, (0, 255, 0), self.thickness)
                
                # 中心点表示
                if show_center and 'center' in person:
                    cv2.circle(annotated, person['center'], 5, (255, 0, 0), -1)
                
                # 顔検出結果の描画
                if show_faces and 'faces' in person:
                    for face in person['faces']:
                        fx1, fy1, fx2, fy2 = face['bbox']
                        
                        # 安定した顔は青、不安定な顔は赤
                        if face.get('stable', False):
                            face_color = (255, 0, 0)  # Blue
                            face_thickness = 2
                        else:
                            face_color = (0, 0, 255)  # Red
                            face_thickness = 1
                        
                        cv2.rectangle(annotated, (fx1, fy1), (fx2, fy2), 
                                    face_color, face_thickness)
                        
                        # トラックID表示
                        track_id = face.get('track_id')
                        if track_id is not None:
                            cv2.putText(annotated, f"ID:{track_id}", (fx1, fy1 - 5),
                                      self.font, 0.4, face_color, 1)
                        
                        # 年齢・性別表示
                        if show_age_gender:
                            age = face.get('age')
                            gender = face.get('gender', 'Unknown')
                            age_range = face.get('age_range', 'Unknown')
                            
                            # デバッグログ
                            if age or gender != 'Unknown':
                                logger.debug(f"Face data - Age: {age}, Gender: {gender}, Range: {age_range}")
                            
                            if age:
                                ag_label = f"{gender}, {age}"
                            else:
                                ag_label = f"{gender}, {age_range}"
                            
                            # ラベル背景
                            label_size, _ = cv2.getTextSize(ag_label, self.font, 0.4, 1)
                            cv2.rectangle(annotated,
                                        (fx1, fy2),
                                        (fx1 + label_size[0], fy2 + label_size[1] + 4),
                                        face_color, -1)
                            
                            # ラベルテキスト
                            cv2.putText(annotated, ag_label, (fx1, fy2 + label_size[1]),
                                      self.font, 0.4, (255, 255, 255), 1)
        
        # 基本的な検出結果の場合（後方互換性）
        elif isinstance(results, list):
            return self.draw_detections(frame, results, show_center, show_confidence)
        
        return annotated


class IntegratedYoloWorker(QThread):
    """統合型YOLOv11検出処理ワーカー（顔検出・年齢性別推定対応）"""
    
    # シグナル定義
    frame_ready = Signal(QImage)
    stats_updated = Signal(dict)
    error_occurred = Signal(str)
    detection_results = Signal(dict)  # 詳細な検出結果
    initialization_progress = Signal(str)  # 初期化進捗の通知
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # 検出関連のコンポーネント
        self.detector = None
        self.pipeline = None
        self.camera = None
        self.visualizer = IntegratedVisualizerEx()
        self.performance = PerformanceMonitor()
        
        # スレッド制御
        self.mutex = QMutex()
        self.wait_condition = QWaitCondition()
        self.is_running = False
        self.is_paused = False
        
        # 基本設定
        self.model_name = "yolo11n.pt"
        self.confidence_threshold = 0.5
        self.camera_index = 0
        self.resolution = (1280, 720)
        self.fps = 30
        self.show_center = False
        self.device = None
        
        # 顔検出・年齢性別推定設定（デフォルトは無効）
        self.enable_face_detection = False  # デフォルトで無効
        self.enable_age_gender = False  # デフォルトで無効
        self.face_confidence = 0.8
        self.use_stable_pipeline = True  # 安定版パイプライン使用
        
        # 統計情報
        self.frame_count = 0
        self.last_stats_time = time.time()
        
    def toggle_face_detection(self, enabled: bool):
        """顔検出のON/OFF切り替え"""
        self.enable_face_detection = enabled
        
        # パイプラインが存在しない、または基本検出器のみの場合は再初期化
        if enabled:
            if not self.pipeline:
                logger.info("Initializing pipeline for face detection")
                self._initialize_pipeline()
            else:
                self.pipeline.update_config(enable_face_detection=enabled)
                logger.info(f"Face detection enabled")
        else:
            if self.pipeline:
                self.pipeline.update_config(enable_face_detection=enabled)
                logger.info(f"Face detection disabled")
    
    def toggle_age_gender(self, enabled: bool):
        """年齢性別推定のON/OFF切り替え"""
        self.enable_age_gender = enabled
        
        # パイプラインが存在しない場合は初期化
        if enabled:
            if not self.pipeline:
                logger.info("Initializing pipeline for age/gender estimation")
                self._initialize_pipeline()
            else:
                self.pipeline.update_config(enable_age_gender=enabled)
                logger.info(f"Age/gender estimation enabled")
        else:
            if self.pipeline:
                self.pipeline.update_config(enable_age_gender=enabled)
                logger.info(f"Age/gender estimation disabled")
    
    def set_face_confidence(self, value: float):
        """顔検出信頼度の設定"""
        self.face_confidence = value
        
        if self.pipeline:
            self.pipeline.update_config(face_confidence=value)
    
    def initialize_components(self):
        """コンポーネントの初期化"""
        try:
            # カメラの初期化
            self.initialization_progress.emit("カメラを初期化中...")
            self.camera = CameraCapture(
                camera_index=self.camera_index,
                resolution=self.resolution,
                fps=self.fps
            )
            
            if not self.camera.open():
                raise RuntimeError("カメラの初期化に失敗しました")
            
            # 検出器の初期化
            if self.enable_face_detection or self.enable_age_gender:
                # 統合パイプライン使用
                self.initialization_progress.emit("AIモデルをロード中...")
                self._initialize_pipeline()
            else:
                # 基本的な人物検出のみ
                self.initialization_progress.emit("人物検出モデルを初期化中...")
                
                # 進捗通知用のコールバック
                def progress_handler(message):
                    self.initialization_progress.emit(message)
                
                self.detector = PersonDetector(
                    model_name=self.model_name,
                    confidence_threshold=self.confidence_threshold,
                    device=self.device,
                    progress_callback=progress_handler
                )
                self.pipeline = None
            
            self.initialization_progress.emit("初期化完了")
            logger.info("検出コンポーネントを初期化しました")
            return True
            
        except Exception as e:
            logger.error(f"初期化エラー: {e}")
            self.error_occurred.emit(str(e))
            return False
    
    def _initialize_pipeline(self):
        """パイプラインの初期化"""
        try:
            # GPU利用可能性をチェック
            try:
                from src.utils.gpu_manager import gpu_manager
                use_gpu = gpu_manager.cuda_available
                logger.info(f"GPU available: {use_gpu}")
            except:
                use_gpu = False
                logger.info("GPU manager not available, using CPU")
            
            # 進捗通知用のコールバック
            def progress_handler(message):
                self.initialization_progress.emit(message)
            
            config = {
                'person_model': self.model_name,
                'person_confidence': self.confidence_threshold,
                'enable_face_detection': self.enable_face_detection,
                'enable_age_gender': self.enable_age_gender,
                'face_confidence': self.face_confidence,
                'face_in_person_only': True,
                'use_gpu': use_gpu,
                'use_advanced_models': True,  # Caffeモデルを使用するため追加
                'progress_callback': progress_handler  # 進捗通知コールバック
            }
            
            if self.use_stable_pipeline:
                self.initialization_progress.emit("検出パイプラインを初期化中...")
                from src.pipelines.stable_detection_pipeline import StableDetectionPipeline
                self.pipeline = StableDetectionPipeline(config)
                logger.info("Stable detection pipeline initialized")
            else:
                from src.pipelines.detection_pipeline import DetectionPipeline
                self.pipeline = DetectionPipeline(config)
                logger.info("Basic detection pipeline initialized")
            
        except Exception as e:
            logger.error(f"Pipeline initialization error: {e}")
            # フォールバック：基本検出器
            self.detector = PersonDetector(
                model_name=self.model_name,
                confidence_threshold=self.confidence_threshold
            )
            self.pipeline = None
    
    def run(self):
        """スレッドのメイン処理"""
        if not self.initialize_components():
            return
        
        self.is_running = True
        
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
                self._process_frame()
                
        except Exception as e:
            logger.error(f"ワーカースレッドエラー: {e}")
            self.error_occurred.emit(str(e))
        finally:
            # クリーンアップ
            self._cleanup()
    
    def _process_frame(self):
        """フレームの処理"""
        ret, frame = self.camera.read()
        if not ret:
            logger.warning("フレームの取得に失敗しました")
            return
        
        try:
            # パフォーマンス計測開始
            self.performance.start_frame()
            
            # 検出処理
            if self.pipeline:
                # パイプライン使用（顔検出あり）
                results = self.pipeline.process_frame(frame)
                detections = results['persons']
                
                # デバッグ: 顔検出結果を確認
                if len(results.get('faces', [])) > 0:
                    for face in results['faces']:
                        if face.get('age') or face.get('gender'):
                            logger.debug(f"Pipeline result - Age: {face.get('age')}, Gender: {face.get('gender')}, Method: {face.get('method')}")
                
                # 詳細結果を送信
                self.detection_results.emit(results)
            else:
                # 基本検出器使用（人物のみ）
                detections = self.detector.detect(frame)
                results = {
                    'persons': detections,
                    'faces': [],
                    'statistics': {
                        'num_persons': len(detections),
                        'num_faces': 0
                    }
                }
            
            # パフォーマンス計測終了
            processing_time = self.performance.end_frame(len(detections))
            
            # 検出結果の描画
            if self.pipeline:
                # 拡張描画（顔・年齢性別含む）
                annotated_frame = self.visualizer.draw_integrated_detections(
                    frame,
                    results,
                    show_center=self.show_center,
                    show_confidence=True,
                    show_faces=self.enable_face_detection,
                    show_age_gender=self.enable_age_gender
                )
            else:
                # 基本描画（人物のみ）
                annotated_frame = self.visualizer.draw_detections(
                    frame,
                    detections,
                    show_center=self.show_center,
                    show_confidence=True
                )
            
            # 統計情報の追加
            stats = self.performance.get_stats()
            info = {
                'Persons': len(detections),
                'Faces': results.get('statistics', {}).get('num_faces', 0),
                'FPS': f"{stats['fps']:.1f}",
                'Time': f"{stats['avg_processing_time']*1000:.1f}ms",
            }
            
            if self.enable_face_detection:
                info['Stable'] = results.get('stability_info', {}).get('stable_faces', 0)
            
            annotated_frame = self.visualizer.draw_info_panel(
                annotated_frame,
                info
            )
            
            # QImageに変換
            qimage = self.convert_cv_to_qimage(annotated_frame)
            
            # シグナルで送信
            if self.is_running:
                self.frame_ready.emit(qimage)
                
                # 統計更新（1秒ごと）
                self.frame_count += 1
                current_time = time.time()
                if current_time - self.last_stats_time > 1.0:
                    self._update_statistics(results, stats)
                    self.last_stats_time = current_time
                
        except Exception as e:
            logger.error(f"フレーム処理エラー: {e}")
    
    def _update_statistics(self, results, perf_stats):
        """統計情報の更新"""
        stats = {
            'fps': perf_stats['fps'],
            'frame_count': self.frame_count,
            'person_count': results.get('statistics', {}).get('num_persons', 0),
            'face_count': results.get('statistics', {}).get('num_faces', 0),
            'avg_person_confidence': results.get('statistics', {}).get('avg_confidence', {}).get('person', 0),
            'avg_face_confidence': results.get('statistics', {}).get('avg_confidence', {}).get('face', 0),
            'gender_distribution': results.get('statistics', {}).get('gender_distribution', {}),
            'age_distribution': results.get('statistics', {}).get('age_distribution', {}),
            'processing_time': perf_stats['avg_processing_time']
        }
        
        # デバッグログ
        if stats['face_count'] > 0:
            logger.debug(f"Stats - Faces: {stats['face_count']}, Gender: {stats['gender_distribution']}, Age: {stats['age_distribution']}")
        
        self.stats_updated.emit(stats)
    
    def convert_cv_to_qimage(self, cv_img):
        """OpenCV画像をQImageに変換"""
        height, width, channel = cv_img.shape
        bytes_per_line = 3 * width
        qimage = QImage(cv_img.data, width, height, bytes_per_line, QImage.Format_RGB888)
        return qimage.rgbSwapped()
    
    def pause(self):
        """検出を一時停止"""
        self.mutex.lock()
        self.is_paused = True
        self.mutex.unlock()
        logger.info("検出を一時停止しました")
    
    def resume(self):
        """検出を再開"""
        self.mutex.lock()
        self.is_paused = False
        self.wait_condition.wakeAll()
        self.mutex.unlock()
        logger.info("検出を再開しました")
    
    def stop(self):
        """検出を停止"""
        self.is_running = False
        self.resume()  # 一時停止中の場合は起こす
        
        if not self.wait(5000):  # 5秒待つ
            logger.warning("ワーカースレッドが正常に停止しませんでした")
            self.terminate()
            self.wait(500)
    
    def _cleanup(self):
        """リソースのクリーンアップ"""
        if self.camera:
            self.camera.release()
        
        if self.pipeline:
            self.pipeline.reset()
        
        logger.info("ワーカーリソースをクリーンアップしました")
    
    def update_threshold(self, threshold: float):
        """信頼度閾値の更新"""
        self.confidence_threshold = threshold
        
        if self.detector:
            self.detector.update_threshold(threshold)
        if self.pipeline:
            self.pipeline.update_config(person_confidence=threshold)
    
    def toggle_center_display(self):
        """中心点表示の切り替え"""
        self.show_center = not self.show_center
    
    def capture_screenshot(self) -> Optional[np.ndarray]:
        """現在のフレームをキャプチャ"""
        if self.camera:
            ret, frame = self.camera.read()
            if ret:
                # 検出処理を実行
                if self.pipeline:
                    results = self.pipeline.process_frame(frame)
                    annotated = self.visualizer.draw_integrated_detections(
                        frame, results,
                        show_center=self.show_center,
                        show_faces=self.enable_face_detection,
                        show_age_gender=self.enable_age_gender
                    )
                elif self.detector:
                    detections = self.detector.detect(frame)
                    annotated = self.visualizer.draw_detections(
                        frame, detections,
                        show_center=self.show_center
                    )
                else:
                    annotated = frame
                
                return annotated
        return None