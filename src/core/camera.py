import cv2
import numpy as np
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class CameraCapture:
    """カメラからの映像取得を管理するクラス"""
    
    def __init__(
        self,
        camera_index: int = 0,
        resolution: Tuple[int, int] = (1280, 720),
        fps: int = 30
    ):
        """
        CameraCaptureの初期化
        
        Args:
            camera_index: カメラデバイスのインデックス
            resolution: 解像度 (width, height)
            fps: フレームレート
        """
        self.camera_index = camera_index
        self.resolution = resolution
        self.fps = fps
        self.cap = None
        self.is_opened = False
        
    def open(self) -> bool:
        """
        カメラを開く
        
        Returns:
            成功した場合True
        """
        try:
            self.cap = cv2.VideoCapture(self.camera_index)
            
            if not self.cap.isOpened():
                logger.warning(f"カメラインデックス {self.camera_index} を開けませんでした")
                
                # 起動時は最大2つのインデックスのみ試す（高速化）
                for i in range(min(2, 5)):
                    if i == self.camera_index:
                        continue
                    logger.info(f"カメラインデックス {i} を試しています...")
                    self.cap = cv2.VideoCapture(i)
                    if self.cap.isOpened():
                        self.camera_index = i
                        logger.info(f"カメラインデックス {i} で接続しました")
                        break
                else:
                    logger.error("利用可能なカメラが見つかりませんでした")
                    return False
            
            self._configure_camera()
            self.is_opened = True
            
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = int(self.cap.get(cv2.CAP_PROP_FPS))
            
            logger.info(f"カメラ設定: {actual_width}x{actual_height} @ {actual_fps}fps")
            return True
            
        except Exception as e:
            logger.error(f"カメラ初期化エラー: {e}")
            return False
    
    def _configure_camera(self) -> None:
        """カメラの設定を行う"""
        if self.cap:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        フレームを読み込む
        
        Returns:
            (成功フラグ, フレーム)
        """
        if not self.is_opened or self.cap is None:
            return False, None
        
        try:
            ret, frame = self.cap.read()
            return ret, frame
        except Exception as e:
            logger.error(f"フレーム読み込みエラー: {e}")
            return False, None
    
    def release(self) -> None:
        """カメラを解放"""
        if self.cap:
            self.cap.release()
            self.is_opened = False
            logger.info("カメラを解放しました")
    
    def get_properties(self) -> dict:
        """
        カメラのプロパティを取得
        
        Returns:
            カメラプロパティの辞書
        """
        if not self.cap:
            return {}
        
        return {
            'width': int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps': int(self.cap.get(cv2.CAP_PROP_FPS)),
            'backend': self.cap.get(cv2.CAP_PROP_BACKEND),
            'fourcc': self.cap.get(cv2.CAP_PROP_FOURCC)
        }
    
    def __enter__(self):
        """コンテキストマネージャーのエントリーポイント"""
        self.open()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """コンテキストマネージャーの終了処理"""
        self.release()