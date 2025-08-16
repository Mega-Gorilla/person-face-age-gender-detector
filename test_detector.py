#!/usr/bin/env python3
"""
検出システムの簡易テストスクリプト
"""

import sys
from pathlib import Path
import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).parent))

from src.core.detector import PersonDetector
from src.ui.visualizer import Visualizer
from src.utils.performance import PerformanceMonitor

def test_detector():
    """検出器のテスト"""
    print("=== 検出器テスト ===")
    
    print("1. PersonDetectorを初期化中...")
    detector = PersonDetector(model_name="yolo11n.pt", confidence_threshold=0.5)
    print("   ✓ 初期化完了")
    
    print("\n2. ダミー画像を作成中...")
    dummy_frame = np.ones((720, 1280, 3), dtype=np.uint8) * 128
    cv2.putText(
        dummy_frame, 
        "Test Frame", 
        (500, 360), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        2, 
        (255, 255, 255), 
        3
    )
    print("   ✓ ダミー画像作成完了")
    
    print("\n3. 検出テスト実行中...")
    detections = detector.detect(dummy_frame)
    print(f"   ✓ 検出完了: {len(detections)}人検出")
    
    print("\n4. Visualizerテスト...")
    visualizer = Visualizer()
    annotated = visualizer.draw_detections(dummy_frame, detections)
    print("   ✓ 可視化完了")
    
    print("\n5. PerformanceMonitorテスト...")
    monitor = PerformanceMonitor()
    monitor.start_frame()
    import time
    time.sleep(0.01)
    monitor.end_frame(len(detections))
    stats = monitor.get_stats()
    print(f"   ✓ FPS: {stats['fps']:.1f}")
    
    print("\n=== テスト完了 ===")
    print("すべてのコンポーネントが正常に動作しています。")
    print("\nカメラを使用した実際のテストを行うには:")
    print("  python main.py")
    
    return True

if __name__ == "__main__":
    try:
        success = test_detector()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\nエラー: {e}")
        sys.exit(1)