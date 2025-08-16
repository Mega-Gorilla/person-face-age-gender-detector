#!/usr/bin/env python3
"""
YOLOv11 リアルタイム人物検出システム
メインエントリーポイント
"""

import sys
import cv2
import logging
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.core.detector import PersonDetector
from src.core.camera import CameraCapture
from src.ui.visualizer import Visualizer
from src.utils.performance import PerformanceMonitor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PersonDetectionApp:
    """人物検出アプリケーションのメインクラス"""
    
    def __init__(self, args):
        """
        アプリケーションの初期化
        
        Args:
            args: コマンドライン引数
        """
        self.args = args
        self.detector = PersonDetector(
            model_name=args.model,
            confidence_threshold=args.confidence,
            device=args.device
        )
        self.camera = CameraCapture(
            camera_index=args.camera,
            resolution=(args.width, args.height),
            fps=args.fps
        )
        self.visualizer = Visualizer()
        self.performance = PerformanceMonitor()
        
        self.paused = False
        self.screenshot_count = 0
    
    def run(self):
        """アプリケーションのメインループ"""
        logger.info("=== YOLOv11 リアルタイム人物検出システム ===")
        
        if not self.camera.open():
            logger.error("カメラの初期化に失敗しました")
            return 1
        
        self._print_instructions()
        
        try:
            while True:
                if not self._process_frame():
                    break
                
                key = cv2.waitKey(1) & 0xFF
                if not self._handle_key(key):
                    break
            
        except KeyboardInterrupt:
            logger.info("キーボード割り込みを検出しました")
        except Exception as e:
            logger.error(f"エラーが発生しました: {e}")
        finally:
            self._cleanup()
        
        return 0
    
    def _process_frame(self) -> bool:
        """
        フレームを処理
        
        Returns:
            処理を続行する場合True
        """
        if self.paused:
            return True
        
        ret, frame = self.camera.read()
        if not ret:
            logger.error("フレームの取得に失敗しました")
            return False
        
        self.performance.start_frame()
        
        detections = self.detector.detect(frame)
        
        processing_time = self.performance.end_frame(len(detections))
        
        annotated_frame = self.visualizer.draw_detections(
            frame, 
            detections,
            show_center=self.args.show_center,
            show_confidence=True
        )
        
        info = {
            '検出人数': len(detections),
            'FPS': f"{self.performance.get_fps():.1f}",
            '処理時間': f"{processing_time*1000:.1f}ms",
            '信頼度閾値': f"{self.detector.confidence_threshold:.2f}"
        }
        
        if self.paused:
            info['状態'] = '一時停止中'
        
        annotated_frame = self.visualizer.draw_info_panel(annotated_frame, info)
        
        cv2.imshow('YOLOv11 Person Detection', annotated_frame)
        
        if self.performance.frame_count % 30 == 0:
            stats = self.performance.get_stats()
            logger.info(self.performance.format_stats(stats))
        
        return True
    
    def _handle_key(self, key: int) -> bool:
        """
        キー入力を処理
        
        Args:
            key: キーコード
            
        Returns:
            処理を続行する場合True
        """
        if key == ord('q') or key == 27:
            logger.info("終了します...")
            return False
        elif key == ord('p'):
            self.paused = not self.paused
            logger.info(f"{'一時停止' if self.paused else '再開'}しました")
        elif key == ord('s'):
            self._save_screenshot()
        elif key == ord('+'):
            self.detector.update_threshold(self.detector.confidence_threshold + 0.05)
        elif key == ord('-'):
            self.detector.update_threshold(self.detector.confidence_threshold - 0.05)
        elif key == ord('r'):
            self.performance.reset()
            logger.info("統計をリセットしました")
        elif key == ord('c'):
            self.args.show_center = not self.args.show_center
            logger.info(f"中心点表示: {'ON' if self.args.show_center else 'OFF'}")
        
        return True
    
    def _save_screenshot(self):
        """スクリーンショットを保存"""
        self.screenshot_count += 1
        filename = f"screenshot_{self.screenshot_count:03d}.jpg"
        
        ret, frame = self.camera.read()
        if ret:
            detections = self.detector.detect(frame)
            annotated_frame = self.visualizer.draw_detections(frame, detections)
            cv2.imwrite(filename, annotated_frame)
            logger.info(f"スクリーンショットを保存しました: {filename}")
    
    def _print_instructions(self):
        """操作方法を表示"""
        print("\n操作方法:")
        print("  - 'q' または 'ESC': 終了")
        print("  - 'p': 一時停止/再開")
        print("  - 's': スクリーンショット保存")
        print("  - '+': 信頼度閾値を上げる")
        print("  - '-': 信頼度閾値を下げる")
        print("  - 'r': 統計をリセット")
        print("  - 'c': 中心点表示のON/OFF")
        print("\n検出を開始します...\n")
    
    def _cleanup(self):
        """終了処理"""
        self.camera.release()
        cv2.destroyAllWindows()
        
        stats = self.performance.get_stats()
        print("\n=== 最終統計 ===")
        print(f"平均FPS: {stats['fps']:.1f}")
        print(f"平均処理時間: {stats['avg_processing_time']*1000:.1f}ms")
        print(f"最小処理時間: {stats['min_processing_time']*1000:.1f}ms")
        print(f"最大処理時間: {stats['max_processing_time']*1000:.1f}ms")
        print(f"総フレーム数: {stats['total_frames']}")
        print(f"総検出数: {stats['total_detections']}")
        print(f"平均検出数: {stats['avg_detections']:.1f}")

def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(
        description='YOLOv11 リアルタイム人物検出システム'
    )
    
    parser.add_argument(
        '--model', '-m',
        default='yolo11n.pt',
        help='使用するYOLOモデル (default: yolo11n.pt)'
    )
    
    parser.add_argument(
        '--confidence', '-c',
        type=float,
        default=0.5,
        help='検出の信頼度閾値 (default: 0.5)'
    )
    
    parser.add_argument(
        '--camera',
        type=int,
        default=0,
        help='カメラデバイスのインデックス (default: 0)'
    )
    
    parser.add_argument(
        '--width',
        type=int,
        default=1280,
        help='カメラの幅 (default: 1280)'
    )
    
    parser.add_argument(
        '--height',
        type=int,
        default=720,
        help='カメラの高さ (default: 720)'
    )
    
    parser.add_argument(
        '--fps',
        type=int,
        default=30,
        help='カメラのFPS (default: 30)'
    )
    
    parser.add_argument(
        '--device',
        choices=['cpu', 'cuda', 'mps'],
        default=None,
        help='実行デバイス (default: auto)'
    )
    
    parser.add_argument(
        '--show-center',
        action='store_true',
        help='検出ボックスの中心点を表示'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='デバッグモードを有効化'
    )
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    app = PersonDetectionApp(args)
    return app.run()

if __name__ == "__main__":
    sys.exit(main())