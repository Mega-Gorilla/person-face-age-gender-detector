#!/usr/bin/env python3
"""
GUI版のテストスクリプト
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

def test_imports():
    """インポートテスト"""
    print("=== GUI モジュールのインポートテスト ===")
    
    try:
        from PySide6 import QtCore, QtWidgets, QtGui
        print(f"✓ PySide6 version: {QtCore.__version__}")
    except ImportError as e:
        print(f"✗ PySide6のインポートに失敗: {e}")
        return False
    
    try:
        from src.gui.windows.main_window import MainWindow
        print("✓ MainWindowのインポート成功")
    except ImportError as e:
        print(f"✗ MainWindowのインポートに失敗: {e}")
        return False
    
    try:
        from src.gui.widgets.video_display import VideoWidget
        print("✓ VideoWidgetのインポート成功")
    except ImportError as e:
        print(f"✗ VideoWidgetのインポートに失敗: {e}")
        return False
    
    try:
        from src.gui.widgets.control_panel import ControlPanel
        print("✓ ControlPanelのインポート成功")
    except ImportError as e:
        print(f"✗ ControlPanelのインポートに失敗: {e}")
        return False
    
    try:
        from src.gui.workers.yolo_worker import YoloDetectionWorker
        print("✓ YoloDetectionWorkerのインポート成功")
    except ImportError as e:
        print(f"✗ YoloDetectionWorkerのインポートに失敗: {e}")
        return False
    
    print("\n=== すべてのGUIモジュールが正常にインポートされました ===")
    return True

def test_gui_creation():
    """GUI作成テスト（ウィンドウは表示しない）"""
    print("\n=== GUI コンポーネントの作成テスト ===")
    
    from PySide6.QtWidgets import QApplication
    
    # アプリケーションを作成（表示はしない）
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    
    try:
        from src.gui.windows.main_window import MainWindow
        window = MainWindow()
        print("✓ MainWindowの作成成功")
        
        # ウィンドウは表示せずに終了
        window.close()
        
        print("\n=== GUIコンポーネントが正常に作成されました ===")
        return True
        
    except Exception as e:
        print(f"✗ GUI作成エラー: {e}")
        return False

def main():
    """メイン関数"""
    print("YOLOv11 GUI版テスト")
    print("=" * 50)
    
    # インポートテスト
    if not test_imports():
        print("\nインポートテストに失敗しました")
        return False
    
    # GUI作成テスト
    if not test_gui_creation():
        print("\nGUI作成テストに失敗しました")
        return False
    
    print("\n" + "=" * 50)
    print("すべてのテストが成功しました！")
    print("\nGUIアプリケーションを実行するには:")
    print("  python gui_main.py")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)