import cv2
import numpy as np
from typing import List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)

class Visualizer:
    """検出結果の可視化クラス"""
    
    def __init__(self):
        """Visualizerの初期化"""
        self.colors = {
            'person': (0, 255, 0),
            'text': (255, 255, 255),
            'background': (0, 0, 0),
            'info': (0, 255, 255),
            'warning': (255, 255, 0),
            'error': (0, 0, 255),
            'center': (255, 0, 0)
        }
        
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.6
        self.thickness = 2
    
    def draw_detections(
        self,
        frame: np.ndarray,
        detections: List[Dict],
        show_center: bool = True,
        show_confidence: bool = True,
        show_boxes: bool = True,
        show_labels: bool = True
    ) -> np.ndarray:
        """
        検出結果を描画
        
        Args:
            frame: 入力フレーム
            detections: 検出結果のリスト
            show_center: 中心点を表示するか
            show_confidence: 信頼度を表示するか
            show_boxes: バウンディングボックスを表示するか
            show_labels: ラベルを表示するか
            
        Returns:
            描画済みフレーム
        """
        annotated_frame = frame.copy()
        
        for i, detection in enumerate(detections):
            if show_boxes:
                self._draw_detection_box(
                    annotated_frame,
                    detection,
                    person_id=i+1,
                    show_confidence=show_confidence,
                    show_label=show_labels
                )
            
            if show_center:
                self._draw_center_point(annotated_frame, detection)
        
        return annotated_frame
    
    def _draw_detection_box(
        self,
        frame: np.ndarray,
        detection: Dict,
        person_id: int,
        show_confidence: bool,
        show_label: bool = True
    ) -> None:
        """
        検出ボックスを描画
        
        Args:
            frame: フレーム
            detection: 検出結果
            person_id: 人物ID
            show_confidence: 信頼度を表示するか
            show_label: ラベルを表示するか
        """
        x1, y1, x2, y2 = detection['bbox']
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), self.colors['person'], 2)
        
        if show_label:
            if show_confidence:
                label = f"Person #{person_id} ({detection['confidence']:.2%})"
            else:
                label = f"Person #{person_id}"
            
            self._draw_label(frame, label, (x1, y1))
    
    def _draw_label(
        self,
        frame: np.ndarray,
        text: str,
        position: Tuple[int, int],
        color: str = 'person'
    ) -> None:
        """
        ラベルを描画
        
        Args:
            frame: フレーム
            text: テキスト
            position: 位置 (x, y)
            color: 色のキー
        """
        x, y = position
        label_size, _ = cv2.getTextSize(text, self.font, self.font_scale, self.thickness)
        label_y = y - 10 if y - 10 > label_size[1] else y + label_size[1] + 10
        
        cv2.rectangle(
            frame,
            (x, label_y - label_size[1] - 5),
            (x + label_size[0] + 5, label_y + 5),
            self.colors[color],
            -1
        )
        
        cv2.putText(
            frame,
            text,
            (x + 2, label_y),
            self.font,
            self.font_scale,
            self.colors['background'],
            self.thickness,
            cv2.LINE_AA
        )
    
    def _draw_center_point(self, frame: np.ndarray, detection: Dict) -> None:
        """
        中心点を描画
        
        Args:
            frame: フレーム
            detection: 検出結果
        """
        cx, cy = detection['center']
        cv2.circle(frame, (cx, cy), 5, self.colors['center'], -1)
    
    def draw_info_panel(
        self,
        frame: np.ndarray,
        info: Dict,
        position: Tuple[int, int] = (10, 30)
    ) -> np.ndarray:
        """
        情報パネルを描画
        
        Args:
            frame: フレーム
            info: 表示する情報の辞書
            position: パネルの位置
            
        Returns:
            描画済みフレーム
        """
        annotated_frame = frame.copy()
        x, y = position
        line_height = 30
        
        for i, (key, value) in enumerate(info.items()):
            text = f"{key}: {value}"
            color = self._get_info_color(key)
            
            cv2.putText(
                annotated_frame,
                text,
                (x, y + i * line_height),
                self.font,
                0.7,
                self.colors[color],
                2,
                cv2.LINE_AA
            )
        
        return annotated_frame
    
    def _get_info_color(self, key: str) -> str:
        """
        情報のキーに基づいて色を決定
        
        Args:
            key: 情報のキー
            
        Returns:
            色のキー
        """
        if 'error' in key.lower():
            return 'error'
        elif 'warning' in key.lower():
            return 'warning'
        elif 'fps' in key.lower() or 'time' in key.lower():
            return 'info'
        else:
            return 'text'
    
    def create_overlay(
        self,
        frame: np.ndarray,
        text: str,
        alpha: float = 0.7
    ) -> np.ndarray:
        """
        オーバーレイテキストを作成
        
        Args:
            frame: フレーム
            text: オーバーレイテキスト
            alpha: 透明度
            
        Returns:
            オーバーレイ済みフレーム
        """
        overlay = frame.copy()
        height, width = frame.shape[:2]
        
        overlay_height = 100
        cv2.rectangle(
            overlay,
            (0, height // 2 - overlay_height // 2),
            (width, height // 2 + overlay_height // 2),
            (0, 0, 0),
            -1
        )
        
        text_size = cv2.getTextSize(text, self.font, 1.5, 3)[0]
        text_x = (width - text_size[0]) // 2
        text_y = (height + text_size[1]) // 2
        
        cv2.putText(
            overlay,
            text,
            (text_x, text_y),
            self.font,
            1.5,
            self.colors['text'],
            3,
            cv2.LINE_AA
        )
        
        return cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)