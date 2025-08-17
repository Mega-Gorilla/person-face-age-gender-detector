"""
映像表示用のカスタムウィジェット
"""

from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QSizePolicy
from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtGui import QImage, QPixmap, QResizeEvent, QPainter
import logging

logger = logging.getLogger(__name__)

class ScaledLabel(QLabel):
    """アスペクト比を保持してスケーリングするカスタムラベル"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._pixmap = None
        
    def setPixmap(self, pixmap):
        """Pixmapを設定（オリジナルを保持）"""
        if isinstance(pixmap, QPixmap):
            self._pixmap = pixmap
            self.update()  # 再描画をトリガー
    
    def paintEvent(self, event):
        """ペイントイベント（スケーリングして描画）"""
        super().paintEvent(event)
        
        if self._pixmap and not self._pixmap.isNull():
            painter = QPainter(self)
            painter.setRenderHint(QPainter.Antialiasing)
            
            # ウィジェットのサイズを取得
            widget_size = self.size()
            
            # アスペクト比を保持してスケール
            scaled_pixmap = self._pixmap.scaled(
                widget_size,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            
            # 中央に配置するための座標を計算
            x = (widget_size.width() - scaled_pixmap.width()) // 2
            y = (widget_size.height() - scaled_pixmap.height()) // 2
            
            # 描画
            painter.drawPixmap(x, y, scaled_pixmap)
    
    def clear(self):
        """表示をクリア"""
        self._pixmap = None
        super().clear()

class VideoWidget(QWidget):
    """映像表示用ウィジェット"""
    
    # ダブルクリックシグナル
    double_clicked = Signal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.is_fullscreen = False
        self._update_pending = False
        
        self.setup_ui()
        
        # フレーム更新レート制限用タイマー
        self.update_timer = QTimer()
        self.update_timer.setInterval(33)  # 約30FPS
        self.update_timer.timeout.connect(self._process_pending_update)
        self.pending_frame = None
    
    def setup_ui(self):
        """UIのセットアップ"""
        # レイアウトの設定
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # カスタム映像表示用ラベル
        self.video_label = ScaledLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("""
            QLabel {
                background-color: #1e1e1e;
                border: 2px solid #333333;
                border-radius: 5px;
            }
        """)
        
        # サイズポリシーを設定（重要）
        self.video_label.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Expanding
        )
        
        # 最小サイズを設定（縮小可能にするため）
        self.video_label.setMinimumSize(320, 240)
        
        # 初期メッセージ
        self.show_message("カメラを初期化中...")
        
        layout.addWidget(self.video_label)
    
    def update_frame(self, qimage: QImage):
        """フレームの更新（レート制限付き）"""
        if qimage is None:
            return
        
        # ペンディングフレームを保存
        self.pending_frame = qimage
        
        # 既に更新がペンディング中でない場合のみ処理
        if not self._update_pending:
            self._update_pending = True
            # 即座に処理（初回）
            self._process_pending_update()
            # タイマーを開始（後続のフレーム用）
            if not self.update_timer.isActive():
                self.update_timer.start()
    
    def _process_pending_update(self):
        """ペンディング中のフレーム更新を処理"""
        if self.pending_frame is not None:
            pixmap = QPixmap.fromImage(self.pending_frame)
            self.video_label.setPixmap(pixmap)
            self.pending_frame = None
        self._update_pending = False
    
    def mouseDoubleClickEvent(self, event):
        """ダブルクリックイベントの処理"""
        if event.button() == Qt.LeftButton:
            self.double_clicked.emit()
    
    def show_message(self, message: str):
        """メッセージの表示"""
        self.video_label.clear()
        self.video_label.setText(message)
        self.video_label.setStyleSheet("""
            QLabel {
                background-color: #1e1e1e;
                border: 2px solid #333333;
                border-radius: 5px;
                color: #ffffff;
                font-size: 18px;
            }
        """)
    
    def show_error(self, error_message: str):
        """エラーメッセージの表示"""
        self.video_label.clear()
        self.video_label.setText(f"エラー: {error_message}")
        self.video_label.setStyleSheet("""
            QLabel {
                background-color: #2a1e1e;
                border: 2px solid #aa3333;
                border-radius: 5px;
                color: #ffaaaa;
                font-size: 16px;
            }
        """)
    
    def clear(self):
        """表示をクリア"""
        self.video_label.clear()
        self.pending_frame = None
        if self.update_timer.isActive():
            self.update_timer.stop()
    
    def closeEvent(self, event):
        """クローズイベント"""
        if self.update_timer.isActive():
            self.update_timer.stop()
        super().closeEvent(event)