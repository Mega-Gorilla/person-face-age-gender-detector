import time
import numpy as np
from typing import List, Dict, Optional
from collections import deque
import logging

logger = logging.getLogger(__name__)

class PerformanceMonitor:
    """パフォーマンス監視クラス"""
    
    def __init__(self, window_size: int = 30):
        """
        PerformanceMonitorの初期化
        
        Args:
            window_size: 統計計算のウィンドウサイズ
        """
        self.window_size = window_size
        self.frame_times = deque(maxlen=window_size)
        self.detection_counts = deque(maxlen=window_size)
        self.start_time = None
        self.frame_count = 0
        self.total_detections = 0
    
    def start_frame(self) -> None:
        """フレーム処理の開始時刻を記録"""
        self.start_time = time.time()
    
    def end_frame(self, detection_count: int = 0) -> float:
        """
        フレーム処理の終了時刻を記録
        
        Args:
            detection_count: 検出された人数
            
        Returns:
            処理時間（秒）
        """
        if self.start_time is None:
            logger.warning("start_frame()が呼ばれていません")
            return 0.0
        
        processing_time = time.time() - self.start_time
        self.frame_times.append(processing_time)
        self.detection_counts.append(detection_count)
        self.frame_count += 1
        self.total_detections += detection_count
        self.start_time = None
        
        return processing_time
    
    def get_stats(self) -> Dict:
        """
        パフォーマンス統計を取得
        
        Returns:
            統計情報の辞書
        """
        if len(self.frame_times) == 0:
            return {
                'fps': 0.0,
                'avg_processing_time': 0.0,
                'min_processing_time': 0.0,
                'max_processing_time': 0.0,
                'avg_detections': 0.0,
                'total_frames': 0,
                'total_detections': 0
            }
        
        times = list(self.frame_times)
        avg_time = np.mean(times)
        
        return {
            'fps': 1.0 / avg_time if avg_time > 0 else 0.0,
            'avg_processing_time': avg_time,
            'min_processing_time': np.min(times),
            'max_processing_time': np.max(times),
            'avg_detections': np.mean(list(self.detection_counts)),
            'total_frames': self.frame_count,
            'total_detections': self.total_detections
        }
    
    def get_fps(self) -> float:
        """
        現在のFPSを取得
        
        Returns:
            FPS値
        """
        if len(self.frame_times) == 0:
            return 0.0
        
        avg_time = np.mean(list(self.frame_times))
        return 1.0 / avg_time if avg_time > 0 else 0.0
    
    def reset(self) -> None:
        """統計をリセット"""
        self.frame_times.clear()
        self.detection_counts.clear()
        self.frame_count = 0
        self.total_detections = 0
        self.start_time = None
        logger.info("パフォーマンス統計をリセットしました")
    
    def format_stats(self, stats: Optional[Dict] = None) -> str:
        """
        統計情報をフォーマット済み文字列として取得
        
        Args:
            stats: 統計情報（Noneの場合は現在の統計を使用）
            
        Returns:
            フォーマット済み文字列
        """
        if stats is None:
            stats = self.get_stats()
        
        return (
            f"FPS: {stats['fps']:.1f} | "
            f"処理時間: {stats['avg_processing_time']*1000:.1f}ms "
            f"(最小: {stats['min_processing_time']*1000:.1f}ms, "
            f"最大: {stats['max_processing_time']*1000:.1f}ms) | "
            f"平均検出数: {stats['avg_detections']:.1f} | "
            f"総フレーム数: {stats['total_frames']}"
        )

class Timer:
    """コンテキストマネージャーとして使用できるタイマー"""
    
    def __init__(self, name: str = "Timer"):
        """
        Timerの初期化
        
        Args:
            name: タイマーの名前
        """
        self.name = name
        self.start_time = None
        self.elapsed_time = None
    
    def __enter__(self):
        """タイマー開始"""
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """タイマー終了"""
        self.elapsed_time = time.time() - self.start_time
        logger.debug(f"{self.name}: {self.elapsed_time*1000:.2f}ms")
    
    def get_elapsed(self) -> float:
        """
        経過時間を取得
        
        Returns:
            経過時間（秒）
        """
        if self.elapsed_time is not None:
            return self.elapsed_time
        elif self.start_time is not None:
            return time.time() - self.start_time
        else:
            return 0.0