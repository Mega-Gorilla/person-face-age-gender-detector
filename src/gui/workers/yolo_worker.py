"""
YOLOv11検出処理ワーカースレッド
"""

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal, QMutex, QWaitCondition
from PySide6.QtGui import QImage
import logging
from typing import Optional, Dict

from src.core.detector import PersonDetector
from src.core.camera import CameraCapture
from src.ui.visualizer import Visualizer
from src.utils.performance import PerformanceMonitor

logger = logging.getLogger(__name__)

class YoloDetectionWorker(QThread):
    """YOLOv11検出処理を行うワーカースレッド"""
    
    # シグナル定義
    frame_ready = Signal(QImage)
    stats_updated = Signal(dict)
    error_occurred = Signal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # 検出関連のコンポーネント
        self.detector = None
        self.camera = None
        self.visualizer = Visualizer()
        self.performance = PerformanceMonitor()
        
        # スレッド制御
        self.mutex = QMutex()
        self.wait_condition = QWaitCondition()
        self.is_running = False
        self.is_paused = False
        
        # 設定
        self.model_name = "yolo11n.pt"
        self.confidence_threshold = 0.5
        self.camera_index = 0
        self.resolution = (1280, 720)
        self.fps = 30
        self.show_center = False
        self.device = None
        
    def initialize_components(self):
        """コンポーネントの初期化"""
        try:
            # 検出器の初期化
            self.detector = PersonDetector(
                model_name=self.model_name,
                confidence_threshold=self.confidence_threshold,
                device=self.device
            )
            
            # カメラの初期化
            self.camera = CameraCapture(
                camera_index=self.camera_index,
                resolution=self.resolution,
                fps=self.fps
            )
            
            if not self.camera.open():
                raise RuntimeError("カメラの初期化に失敗しました")
                
            logger.info("検出コンポーネントを初期化しました")
            return True
            
        except Exception as e:
            logger.error(f"初期化エラー: {e}")
            self.error_occurred.emit(str(e))
            return False
    
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
            
            # 人物検出
            detections = self.detector.detect(frame)
            
            # パフォーマンス計測終了
            processing_time = self.performance.end_frame(len(detections))
            
            # 検出結果の描画
            annotated_frame = self.visualizer.draw_detections(
                frame,
                detections,
                show_center=self.show_center,
                show_confidence=True
            )
            
            # 統計情報の追加
            stats = self.performance.get_stats()
            info = {
                'Detected': len(detections),
                'FPS': f"{stats['fps']:.1f}",
                'Time': f"{stats['avg_processing_time']*1000:.1f}ms",
                'Threshold': f"{self.confidence_threshold:.2f}"
            }
            
            annotated_frame = self.visualizer.draw_info_panel(
                annotated_frame,
                info
            )
            
            # QImageに変換
            qimage = self.convert_cv_to_qimage(annotated_frame)
            
            # シグナルで送信
            if self.is_running:  # 終了中でないことを確認
                self.frame_ready.emit(qimage)
                self.stats_updated.emit({
                    'fps': stats['fps'],
                    'processing_time': stats['avg_processing_time'],
                    'detection_count': len(detections),
                    'confidence_threshold': self.confidence_threshold,
                    'total_frames': stats['total_frames'],
                    'total_detections': stats['total_detections']
                })
                
        except Exception as e:
            logger.error(f"フレーム処理エラー: {e}")
            if self.is_running:
                self.error_occurred.emit(str(e))
    
    def convert_cv_to_qimage(self, cv_img: np.ndarray) -> QImage:
        """OpenCV画像をQImageに変換"""
        height, width, channel = cv_img.shape
        bytes_per_line = 3 * width
        
        # BGRからRGBに変換
        rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        
        # QImageを作成
        qimage = QImage(
            rgb_img.data,
            width,
            height,
            bytes_per_line,
            QImage.Format_RGB888
        )
        
        return qimage.copy()
    
    def pause(self):
        """一時停止"""
        self.mutex.lock()
        self.is_paused = True
        self.mutex.unlock()
        logger.info("検出を一時停止しました")
    
    def resume(self):
        """再開"""
        self.mutex.lock()
        self.is_paused = False
        self.wait_condition.wakeAll()
        self.mutex.unlock()
        logger.info("検出を再開しました")
    
    def stop(self):
        """停止"""
        self.is_running = False
        
        # 一時停止中の場合は再開
        if self.is_paused:
            self.resume()
        
        # スレッドの終了を待つ（最大1秒）
        if not self.wait(1000):
            logger.warning("ワーカースレッドの正常終了がタイムアウトしました")
            self.terminate()  # 強制終了
            self.wait(500)   # 少し待つ
        
        logger.info("検出スレッドを停止しました")
    
    def _cleanup(self):
        """リソースのクリーンアップ"""
        if self.camera:
            self.camera.release()
            self.camera = None
        
        self.detector = None
        logger.info("リソースをクリーンアップしました")
    
    def update_confidence_threshold(self, value: float):
        """信頼度閾値の更新"""
        self.confidence_threshold = value
        if self.detector:
            self.detector.update_threshold(value)
        logger.info(f"信頼度閾値を更新: {value:.2f}")
    
    def update_model(self, model_name: str):
        """モデルの更新"""
        self.model_name = model_name
        if self.detector:
            try:
                self.detector = PersonDetector(
                    model_name=model_name,
                    confidence_threshold=self.confidence_threshold,
                    device=self.device
                )
                logger.info(f"モデルを更新: {model_name}")
            except Exception as e:
                logger.error(f"モデル更新エラー: {e}")
                self.error_occurred.emit(str(e))
    
    def toggle_center_display(self):
        """中心点表示の切り替え"""
        self.show_center = not self.show_center
        logger.info(f"中心点表示: {'ON' if self.show_center else 'OFF'}")
    
    def reset_stats(self):
        """統計のリセット"""
        self.performance.reset()
        logger.info("統計をリセットしました")
    
    def capture_screenshot(self) -> Optional[np.ndarray]:
        """スクリーンショットの撮影"""
        if self.camera:
            ret, frame = self.camera.read()
            if ret:
                detections = self.detector.detect(frame)
                annotated_frame = self.visualizer.draw_detections(
                    frame,
                    detections,
                    show_center=self.show_center,
                    show_confidence=True
                )
                return annotated_frame
        return None