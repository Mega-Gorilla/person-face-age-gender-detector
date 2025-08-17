"""
メインウィンドウクラス
"""

import cv2
from datetime import datetime
from pathlib import Path
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QMessageBox, QFileDialog, QSplitter, QStatusBar,
    QTabWidget
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction, QKeySequence
import logging

from src.gui.widgets.video_display import VideoWidget
from src.gui.widgets.control_panel import ControlPanel
from src.gui.widgets.file_processor import FileProcessorWidget
from src.gui.workers.yolo_worker import YoloDetectionWorker
from src.gui.workers.file_worker import FileProcessingWorker
from src.utils.version import (
    APP_NAME, VERSION_STRING, get_about_text,
    get_full_version
)

logger = logging.getLogger(__name__)

class MainWindow(QMainWindow):
    """Main window"""
    
    def __init__(self):
        super().__init__()
        
        self.detection_worker = None
        self.file_worker = None
        self.screenshot_count = 0
        
        self.setup_ui()
        self.setup_menu()
        self.setup_connections()
        self.setup_detection_worker()
        self.setup_file_worker()
    
    def setup_ui(self):
        """Setup UI"""
        self.setWindowTitle(get_full_version())
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
        
        # Central widget with tabs
        self.tab_widget = QTabWidget()
        self.setCentralWidget(self.tab_widget)
        
        # Stream tab (Webcam mode)
        self.stream_tab = QWidget()
        self.setup_stream_tab()
        self.tab_widget.addTab(self.stream_tab, "🎥 Stream")
        
        # File tab (Video file processing mode)
        self.file_tab = FileProcessorWidget()
        self.tab_widget.addTab(self.file_tab, "📁 File")
        
        # Set tab style
        self.tab_widget.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #555555;
                background-color: #2b2b2b;
            }
            QTabBar::tab {
                background-color: #363636;
                color: #ffffff;
                padding: 8px 16px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background-color: #2196F3;
            }
            QTabBar::tab:hover {
                background-color: #454545;
            }
        """)
        
    def setup_stream_tab(self):
        """Setup stream tab layout"""
        # Main layout for stream tab
        main_layout = QHBoxLayout(self.stream_tab)
        
        # Splitter
        splitter = QSplitter(Qt.Horizontal)
        
        # Left side: Video display
        self.video_widget = VideoWidget()
        splitter.addWidget(self.video_widget)
        
        # Right side: Control panel
        self.control_panel = ControlPanel()
        self.control_panel.setMaximumWidth(350)
        splitter.addWidget(self.control_panel)
        
        # Initial splitter ratio
        splitter.setSizes([1050, 350])
        
        main_layout.addWidget(splitter)
        
        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")
    
    def setup_menu(self):
        """Setup menu bar"""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("File")
        
        # Save screenshot
        screenshot_action = QAction("Save Screenshot", self)
        screenshot_action.setShortcut(QKeySequence("Ctrl+S"))
        screenshot_action.triggered.connect(self.save_screenshot)
        file_menu.addAction(screenshot_action)
        
        file_menu.addSeparator()
        
        # Exit
        exit_action = QAction("Exit", self)
        exit_action.setShortcut(QKeySequence("Ctrl+Q"))
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # View menu
        view_menu = menubar.addMenu("View")
        
        # Fullscreen
        fullscreen_action = QAction("Fullscreen", self)
        fullscreen_action.setShortcut(QKeySequence("F11"))
        fullscreen_action.setCheckable(True)
        fullscreen_action.triggered.connect(self.toggle_fullscreen)
        view_menu.addAction(fullscreen_action)
        
        # Help menu
        help_menu = menubar.addMenu("Help")
        
        # About
        about_action = QAction("About", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
    
    def setup_connections(self):
        """Setup signal-slot connections"""
        # Control panel signals
        self.control_panel.play_pause_clicked.connect(self.toggle_detection)
        self.control_panel.screenshot_clicked.connect(self.save_screenshot)
        self.control_panel.confidence_changed.connect(self.update_confidence)
        self.control_panel.model_changed.connect(self.update_model)
        self.control_panel.center_display_toggled.connect(self.toggle_center_display)
        self.control_panel.reset_stats_clicked.connect(self.reset_statistics)
        self.control_panel.camera_settings_changed.connect(self.update_camera_settings)
        
        # Video widget signals
        self.video_widget.double_clicked.connect(self.toggle_fullscreen)
        
        # File processor signals (File tab)
        self.file_tab.process_started.connect(self.on_file_processing_started)
        self.file_tab.process_stopped.connect(self.on_file_processing_stopped)
    
    def setup_detection_worker(self):
        """Setup detection worker for stream mode"""
        self.detection_worker = YoloDetectionWorker(self)
        
        # Connect signals
        self.detection_worker.frame_ready.connect(self.video_widget.update_frame)
        self.detection_worker.stats_updated.connect(self.control_panel.update_statistics)
        self.detection_worker.error_occurred.connect(self.handle_error)
        
        # Start worker
        self.detection_worker.start()
        self.status_bar.showMessage("Detection started")
    
    def setup_file_worker(self):
        """Setup file processing worker"""
        self.file_worker = FileProcessingWorker(self)
        
        # Connect signals
        self.file_worker.progress_updated.connect(self.file_tab.update_progress)
        self.file_worker.frame_processed.connect(self.on_frame_processed)
        self.file_worker.processing_completed.connect(self.file_tab.processing_completed)
        self.file_worker.error_occurred.connect(self.handle_file_error)
        self.file_worker.log_message.connect(self.file_tab.log_message)
    
    def toggle_detection(self):
        """Toggle detection pause/resume"""
        if self.detection_worker:
            if self.control_panel.is_playing:
                self.detection_worker.resume()
                self.status_bar.showMessage("Detection resumed")
            else:
                self.detection_worker.pause()
                self.status_bar.showMessage("Detection paused")
    
    def save_screenshot(self):
        """Save screenshot"""
        if not self.detection_worker:
            return
        
        # Capture screenshot
        frame = self.detection_worker.capture_screenshot()
        if frame is None:
            QMessageBox.warning(self, "Warning", "Failed to capture screenshot")
            return
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_filename = f"screenshot_{timestamp}.jpg"
        
        # Save dialog
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Save Screenshot",
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
            # Stop detection worker
            if self.detection_worker:
                self.detection_worker.stop()
                self.detection_worker = None
            
            # Stop file worker
            if self.file_worker:
                self.file_worker.stop()
                self.file_worker = None
            
            event.accept()
            logger.info("Application closed")
        else:
            event.ignore()
    
    def on_file_processing_started(self, params: dict):
        """Handle file processing start"""
        if not self.file_worker:
            self.setup_file_worker()
        
        self.file_worker.set_parameters(params)
        self.file_worker.start()
        self.status_bar.showMessage("File processing started")
    
    def on_file_processing_stopped(self):
        """Handle file processing stop"""
        if self.file_worker:
            self.file_worker.stop()
        self.status_bar.showMessage("File processing stopped")
    
    def on_frame_processed(self, frame_num: int, timestamp: float, detections: list):
        """Handle frame processed event"""
        # Update results table
        if detections:
            avg_conf = sum(d.get('confidence', 0) for d in detections) / len(detections)
            objects = f"{len(detections)} person(s)"
        else:
            avg_conf = 0.0
            objects = "No detections"
        
        self.file_tab.add_result_row(frame_num, timestamp, len(detections), avg_conf, objects)
    
    def handle_file_error(self, error_message: str):
        """Handle file processing error"""
        logger.error(f"File processing error: {error_message}")
        QMessageBox.critical(self, "Error", f"File processing error: {error_message}")
        self.status_bar.showMessage(f"Error: {error_message}")