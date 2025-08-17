#!/usr/bin/env python3
"""
YOLOv11 人物検出システム GUI版
メインエントリーポイント
"""

import sys
import logging
from pathlib import Path
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent))

from src.gui.windows.main_window import MainWindow

# ロギングの設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """メイン関数"""
    # アプリケーションの作成
    app = QApplication(sys.argv)
    
    # アプリケーション設定
    app.setApplicationName("YOLOv11 Person Detector")
    app.setOrganizationName("YOLOv11 Team")
    
    # High DPI対応
    app.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    app.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    
    # スタイル設定
    app.setStyle("Fusion")
    
    # メインウィンドウの作成と表示
    try:
        window = MainWindow()
        window.show()
        
        logger.info("GUI版 YOLOv11 人物検出システムを起動しました")
        
        # アプリケーションの実行
        sys.exit(app.exec())
        
    except Exception as e:
        logger.error(f"起動エラー: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()