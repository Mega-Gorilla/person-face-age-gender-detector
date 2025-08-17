"""
メインウィンドウクラス
"""

import cv2
from datetime import datetime
from pathlib import Path
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QMessageBox, QFileDialog, QSplitter, QStatusBar
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction, QKeySequence
import logging

from src.gui.widgets.video_display import VideoWidget
from src.gui.widgets.control_panel import ControlPanel
from src.gui.workers.yolo_worker import YoloDetectionWorker

logger = logging.getLogger(__name__)

class MainWindow(QMainWindow):
    """メインウィンドウ"""
    
    def __init__(self):
        super().__init__()
        
        self.detection_worker = None
        self.screenshot_count = 0
        
        self.setup_ui()
        self.setup_menu()
        self.setup_connections()
        self.setup_detection_worker()
    
    def setup_ui(self):
        """UIのセットアップ"""
        self.setWindowTitle("YOLOv11 人物検出システム - GUI版")
        self.setGeometry(100, 100, 1400, 900)
        
        # スタイルシートの設定
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2b2b2b;
            }
            QGroupBox {
                background-color: #363636;
                border: 1px solid #555555;
                border-radius: 5px;
                margin-top: 10px;
                color: #ffffff;
                font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            QLabel {
                color: #ffffff;
            }
            QComboBox, QSpinBox {
                background-color: #454545;
                color: #ffffff;
                border: 1px solid #555555;
                border-radius: 3px;
                padding: 2px;
            }
            QSlider::groove:horizontal {
                height: 6px;
                background: #454545;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #2196F3;
                width: 18px;
                height: 18px;
                border-radius: 9px;
                margin: -6px 0;
            }
            QCheckBox {
                color: #ffffff;
            }
        """)
        
        # 中央ウィジェット
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # メインレイアウト
        main_layout = QHBoxLayout(central_widget)
        
        # スプリッター
        splitter = QSplitter(Qt.Horizontal)
        
        # 左側: 映像表示
        self.video_widget = VideoWidget()
        splitter.addWidget(self.video_widget)
        
        # 右側: コントロールパネル
        self.control_panel = ControlPanel()
        self.control_panel.setMaximumWidth(350)
        splitter.addWidget(self.control_panel)
        
        # スプリッターの初期比率
        splitter.setSizes([1050, 350])
        
        main_layout.addWidget(splitter)
        
        # ステータスバー
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("準備完了")
    
    def setup_menu(self):
        """メニューバーのセットアップ"""
        menubar = self.menuBar()
        
        # ファイルメニュー
        file_menu = menubar.addMenu("ファイル")
        
        # スクリーンショット保存
        screenshot_action = QAction("スクリーンショット保存", self)
        screenshot_action.setShortcut(QKeySequence("Ctrl+S"))
        screenshot_action.triggered.connect(self.save_screenshot)
        file_menu.addAction(screenshot_action)
        
        file_menu.addSeparator()
        
        # 終了
        exit_action = QAction("終了", self)
        exit_action.setShortcut(QKeySequence("Ctrl+Q"))
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # 表示メニュー
        view_menu = menubar.addMenu("表示")
        
        # フルスクリーン
        fullscreen_action = QAction("フルスクリーン", self)
        fullscreen_action.setShortcut(QKeySequence("F11"))
        fullscreen_action.setCheckable(True)
        fullscreen_action.triggered.connect(self.toggle_fullscreen)
        view_menu.addAction(fullscreen_action)
        
        # ヘルプメニュー
        help_menu = menubar.addMenu("ヘルプ")
        
        # バージョン情報
        about_action = QAction("バージョン情報", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
    
    def setup_connections(self):
        """シグナル・スロットの接続"""
        # コントロールパネルのシグナル
        self.control_panel.play_pause_clicked.connect(self.toggle_detection)
        self.control_panel.screenshot_clicked.connect(self.save_screenshot)
        self.control_panel.confidence_changed.connect(self.update_confidence)
        self.control_panel.model_changed.connect(self.update_model)
        self.control_panel.center_display_toggled.connect(self.toggle_center_display)
        self.control_panel.reset_stats_clicked.connect(self.reset_statistics)
        self.control_panel.camera_settings_changed.connect(self.update_camera_settings)
        
        # ビデオウィジェットのシグナル
        self.video_widget.double_clicked.connect(self.toggle_fullscreen)
    
    def setup_detection_worker(self):
        """検出ワーカーのセットアップ"""
        self.detection_worker = YoloDetectionWorker(self)
        
        # シグナルの接続
        self.detection_worker.frame_ready.connect(self.video_widget.update_frame)
        self.detection_worker.stats_updated.connect(self.control_panel.update_statistics)
        self.detection_worker.error_occurred.connect(self.handle_error)
        
        # ワーカーの開始
        self.detection_worker.start()
        self.status_bar.showMessage("検出を開始しました")
    
    def toggle_detection(self):
        """検出の一時停止/再開"""
        if self.detection_worker:
            if self.control_panel.is_playing:
                self.detection_worker.resume()
                self.status_bar.showMessage("検出を再開しました")
            else:
                self.detection_worker.pause()
                self.status_bar.showMessage("検出を一時停止しました")
    
    def save_screenshot(self):
        """スクリーンショットの保存"""
        if not self.detection_worker:
            return
        
        # スクリーンショットの取得
        frame = self.detection_worker.capture_screenshot()
        if frame is None:
            QMessageBox.warning(self, "警告", "スクリーンショットの取得に失敗しました")
            return
        
        # ファイル名の生成
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_filename = f"screenshot_{timestamp}.jpg"
        
        # 保存ダイアログ
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "スクリーンショットを保存",
            default_filename,
            "画像ファイル (*.jpg *.png)"
        )
        
        if filename:
            # 画像の保存
            cv2.imwrite(filename, frame)
            self.screenshot_count += 1
            self.status_bar.showMessage(f"スクリーンショットを保存しました: {filename}")
            logger.info(f"スクリーンショットを保存: {filename}")
    
    def update_confidence(self, value: float):
        """信頼度閾値の更新"""
        if self.detection_worker:
            self.detection_worker.update_confidence_threshold(value)
            self.status_bar.showMessage(f"信頼度閾値を更新: {value:.2f}")
    
    def update_model(self, model_name: str):
        """モデルの更新"""
        if self.detection_worker:
            self.detection_worker.update_model(model_name)
            self.status_bar.showMessage(f"モデルを更新: {model_name}")
    
    def toggle_center_display(self, checked: bool):
        """中心点表示の切り替え"""
        if self.detection_worker:
            self.detection_worker.toggle_center_display()
            self.status_bar.showMessage(f"中心点表示: {'ON' if checked else 'OFF'}")
    
    def reset_statistics(self):
        """統計のリセット"""
        if self.detection_worker:
            self.detection_worker.reset_stats()
            self.status_bar.showMessage("統計をリセットしました")
    
    def update_camera_settings(self, settings: dict):
        """カメラ設定の更新"""
        reply = QMessageBox.question(
            self,
            "確認",
            "カメラ設定を変更すると検出が一時的に停止します。続行しますか？",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            # 現在のワーカーを停止
            if self.detection_worker:
                self.detection_worker.stop()
            
            # 新しい設定でワーカーを再作成
            self.detection_worker = YoloDetectionWorker(self)
            self.detection_worker.camera_index = settings['camera_index']
            self.detection_worker.resolution = (settings['width'], settings['height'])
            self.detection_worker.fps = settings['fps']
            
            # シグナルの再接続
            self.detection_worker.frame_ready.connect(self.video_widget.update_frame)
            self.detection_worker.stats_updated.connect(self.control_panel.update_statistics)
            self.detection_worker.error_occurred.connect(self.handle_error)
            
            # ワーカーの開始
            self.detection_worker.start()
            self.status_bar.showMessage("カメラ設定を更新しました")
    
    def toggle_fullscreen(self):
        """フルスクリーンの切り替え"""
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()
    
    def handle_error(self, error_message: str):
        """エラーの処理"""
        logger.error(f"エラー: {error_message}")
        self.video_widget.show_error(error_message)
        self.status_bar.showMessage(f"エラー: {error_message}")
    
    def show_about(self):
        """バージョン情報の表示"""
        QMessageBox.about(
            self,
            "バージョン情報",
            "YOLOv11 人物検出システム GUI版\n\n"
            "Version: 2.0.0\n"
            "Framework: PySide6 + YOLOv11\n"
            "Author: YOLOv11 Development Team\n\n"
            "最新のYOLOv11モデルを使用した\n"
            "リアルタイム人物検出システムです。"
        )
    
    def closeEvent(self, event):
        """終了イベントの処理"""
        reply = QMessageBox.question(
            self,
            "確認",
            "アプリケーションを終了しますか？",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            # 検出ワーカーの停止
            if self.detection_worker:
                self.detection_worker.stop()
                self.detection_worker = None
            
            event.accept()
            logger.info("アプリケーションを終了しました")
        else:
            event.ignore()