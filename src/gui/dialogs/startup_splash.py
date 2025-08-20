"""
起動時のスプラッシュスクリーン
"""

from PySide6.QtWidgets import QSplashScreen, QApplication
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPixmap, QPainter, QFont, QColor
import logging

logger = logging.getLogger(__name__)


class StartupSplash(QSplashScreen):
    """起動時のスプラッシュスクリーン"""
    
    def __init__(self):
        # スプラッシュ用の画像を作成（シンプルな背景）
        pixmap = QPixmap(500, 300)
        pixmap.fill(QColor(43, 43, 43))  # ダークグレー背景
        
        # テキストを描画
        painter = QPainter(pixmap)
        painter.setPen(QColor(255, 255, 255))
        
        # タイトル
        title_font = QFont("Arial", 20, QFont.Bold)
        painter.setFont(title_font)
        painter.drawText(pixmap.rect(), Qt.AlignTop | Qt.AlignHCenter, 
                        "\n\nYOLOv11 人物検出システム")
        
        # バージョン
        version_font = QFont("Arial", 12)
        painter.setFont(version_font)
        painter.drawText(pixmap.rect(), Qt.AlignCenter,
                        "Version 2.3.0\n\nPowered by YOLOv11 & PySide6")
        
        # 初期メッセージ
        painter.drawText(pixmap.rect(), Qt.AlignBottom | Qt.AlignHCenter,
                        "\n\n\n\n\n\nシステムを起動中...\n\n")
        
        painter.end()
        
        super().__init__(pixmap)
        self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint)
        
    def update_message(self, message: str):
        """メッセージを更新"""
        # より詳細なメッセージに応じて色を変更
        color = QColor(0, 255, 0)  # デフォルトは緑
        
        if "YOLO" in message or "人物検出" in message:
            color = QColor(255, 200, 0)  # オレンジ
        elif "顔検出" in message:
            color = QColor(0, 200, 255)  # 水色
        elif "年齢・性別" in message:
            color = QColor(255, 100, 255)  # ピンク
        elif "カメラ" in message:
            color = QColor(100, 255, 100)  # 明るい緑
        elif "完了" in message:
            color = QColor(0, 255, 0)  # 緑
            
        self.showMessage(message, 
                         Qt.AlignBottom | Qt.AlignHCenter, 
                         color)
        QApplication.processEvents()
        
    def show_and_process(self):
        """スプラッシュを表示して処理を進める"""
        self.show()
        QApplication.processEvents()