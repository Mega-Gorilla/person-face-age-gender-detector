#!/usr/bin/env python3
"""
YOLOv11 人物検出システム GUI版
メインエントリーポイント

このモジュールはGUIアプリケーションの起動点として機能し、
必要なモデルのチェックとダウンロード、メインウィンドウの表示を行います。
"""

import sys
import logging
from pathlib import Path
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt, QCoreApplication
from PySide6.QtGui import QIcon

# プロジェクトルートをパスに追加（相対インポートを可能にするため）
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.gui.windows.main_window import MainWindow
from src.gui.dialogs.model_download_dialog import check_and_download_models
from src.core.age_gender_caffe import check_gdown_installed

# ロギングの設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def setup_application():
    """アプリケーションの初期設定を行う
    
    Returns:
        QApplication: 設定済みのアプリケーションインスタンス
    """
    # High DPI設定（Qt6では自動的に有効）
    # Qt5との互換性のためにQCoreApplicationで設定
    QCoreApplication.setAttribute(Qt.AA_ShareOpenGLContexts)
    
    # アプリケーションの作成
    app = QApplication(sys.argv)
    
    # アプリケーション設定
    app.setApplicationName("YOLOv11 人物検出システム")
    app.setOrganizationName("PersonDetector")
    app.setApplicationDisplayName("YOLOv11 人物・顔検出 GUI")
    
    # スタイル設定（モダンな外観）
    app.setStyle("Fusion")
    
    # ウィンドウアイコンの設定（アイコンファイルが存在する場合）
    icon_path = project_root / "assets" / "icon.png"
    if icon_path.exists():
        app.setWindowIcon(QIcon(str(icon_path)))
    
    return app


def main():
    """メイン関数"""
    # アプリケーションのセットアップ
    app = setup_application()
    
    # メインウィンドウの作成と表示
    try:
        # 年齢・性別推定モデルのチェックとダウンロード
        logger.info("年齢・性別推定モデルをチェック中...")
        
        # gdownパッケージの確認
        if not check_gdown_installed():
            logger.warning("gdownがインストールされていません。年齢・性別推定機能が制限されます。")
            logger.warning("インストール方法: pip install gdown")
        else:
            # モデルのチェックとダウンロード（必要に応じてダイアログ表示）
            models_ready = check_and_download_models()
            if models_ready:
                logger.info("年齢・性別推定モデルの準備完了")
            else:
                logger.warning("年齢・性別推定モデルが利用できません - 機能は無効化されます")
        
        # メインウィンドウの作成
        window = MainWindow()
        window.show()
        
        logger.info("GUI版 YOLOv11 人物検出システムを起動しました")
        
        # アプリケーションの実行
        return app.exec()
        
    except ImportError as e:
        logger.error(f"モジュールのインポートエラー: {e}")
        logger.error("必要なパッケージがインストールされているか確認してください")
        logger.error("インストール方法: pip install -r requirements.txt")
        return 1
    except FileNotFoundError as e:
        logger.error(f"ファイルが見つかりません: {e}")
        return 1
    except Exception as e:
        logger.error(f"予期しない起動エラー: {e}")
        logger.error(f"エラーの詳細: {type(e).__name__}")
        import traceback
        logger.debug(traceback.format_exc())
        return 1

if __name__ == "__main__":
    sys.exit(main())