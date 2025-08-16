import numpy as np
from ultralytics import YOLO
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)

class PersonDetector:
    """YOLOv11を使用した人物検出クラス"""
    
    PERSON_CLASS_ID = 0
    
    def __init__(
        self, 
        model_name: str = "yolo11n.pt",
        confidence_threshold: float = 0.5,
        device: Optional[str] = None
    ):
        """
        PersonDetectorの初期化
        
        Args:
            model_name: 使用するYOLOモデル名
            confidence_threshold: 検出の信頼度閾値
            device: 実行デバイス ('cpu', 'cuda', None=auto)
        """
        self.confidence_threshold = confidence_threshold
        self.device = device
        
        try:
            logger.info(f"モデル {model_name} を読み込み中...")
            self.model = YOLO(model_name)
            if self.device:
                self.model.to(self.device)
            logger.info(f"モデルの読み込みが完了しました")
        except Exception as e:
            logger.error(f"モデル読み込みエラー: {e}")
            logger.info("モデルをダウンロード中...")
            self.model = YOLO(model_name)
            if self.device:
                self.model.to(self.device)
    
    def detect(self, frame: np.ndarray) -> List[Dict]:
        """
        フレームから人物を検出
        
        Args:
            frame: 入力画像フレーム
            
        Returns:
            検出結果のリスト
        """
        try:
            results = self.model(frame, verbose=False, device=self.device)
            return self._process_results(results)
        except Exception as e:
            logger.error(f"検出エラー: {e}")
            return []
    
    def _process_results(self, results) -> List[Dict]:
        """
        YOLOの結果を処理して人物検出結果を抽出
        
        Args:
            results: YOLOの検出結果
            
        Returns:
            人物検出結果のリスト
        """
        person_detections = []
        
        for result in results:
            if result.boxes is None:
                continue
                
            for box in result.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                
                if class_id == self.PERSON_CLASS_ID and confidence >= self.confidence_threshold:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    detection = {
                        'bbox': (x1, y1, x2, y2),
                        'confidence': confidence,
                        'center': ((x1 + x2) // 2, (y1 + y2) // 2),
                        'area': (x2 - x1) * (y2 - y1),
                        'width': x2 - x1,
                        'height': y2 - y1
                    }
                    person_detections.append(detection)
        
        return person_detections
    
    def update_threshold(self, new_threshold: float) -> None:
        """
        信頼度閾値を更新
        
        Args:
            new_threshold: 新しい信頼度閾値
        """
        self.confidence_threshold = max(0.0, min(1.0, new_threshold))
        logger.info(f"信頼度閾値を更新: {self.confidence_threshold:.2f}")