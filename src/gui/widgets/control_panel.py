"""
コントロールパネルウィジェット
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QPushButton, QSlider, QLabel, QComboBox,
    QCheckBox, QSpinBox, QGridLayout, QMessageBox
)
from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtGui import QFont
import logging
from src.utils.camera_detector import get_available_cameras

logger = logging.getLogger(__name__)

class ControlPanel(QWidget):
    """コントロールパネル"""
    
    # シグナル定義
    play_pause_clicked = Signal()
    screenshot_clicked = Signal()
    confidence_changed = Signal(float)
    model_changed = Signal(str)
    center_display_toggled = Signal(bool)
    reset_stats_clicked = Signal()
    camera_settings_changed = Signal(dict)
    
    # 顔検出関連のシグナル
    face_detection_toggled = Signal(bool)
    age_gender_toggled = Signal(bool)
    face_confidence_changed = Signal(float)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.is_playing = True
        self.available_cameras = []  # 利用可能なカメラリスト
        self.setup_ui()
        
        # カメラリストを初期化（非同期で実行、起動を高速化）
        QTimer.singleShot(100, self.quick_camera_init)
    
    def setup_ui(self):
        """UIのセットアップ"""
        layout = QVBoxLayout(self)
        
        # メインコントロール
        layout.addWidget(self.create_main_controls())
        
        # 検出設定
        layout.addWidget(self.create_detection_settings())
        
        # 顔検出設定
        layout.addWidget(self.create_face_detection_settings())
        
        # カメラ設定
        layout.addWidget(self.create_camera_settings())
        
        # 統計情報
        layout.addWidget(self.create_statistics_display())
        
        # ストレッチを追加
        layout.addStretch()
    
    def create_main_controls(self) -> QGroupBox:
        """メインコントロールの作成"""
        group = QGroupBox("Main Controls")
        layout = QVBoxLayout()
        
        # 再生/一時停止ボタン
        self.play_pause_btn = QPushButton("⏸ Pause")
        self.play_pause_btn.setMinimumHeight(40)
        self.play_pause_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                border-radius: 5px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QPushButton:pressed {
                background-color: #0D47A1;
            }
        """)
        self.play_pause_btn.clicked.connect(self.on_play_pause_clicked)
        layout.addWidget(self.play_pause_btn)
        
        # スクリーンショットボタン
        self.screenshot_btn = QPushButton("📷 Screenshot")
        self.screenshot_btn.setMinimumHeight(35)
        self.screenshot_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 5px;
                font-size: 13px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        self.screenshot_btn.clicked.connect(self.screenshot_clicked.emit)
        layout.addWidget(self.screenshot_btn)
        
        # 統計リセットボタン
        self.reset_stats_btn = QPushButton("🔄 Reset Stats")
        self.reset_stats_btn.setMinimumHeight(30)
        self.reset_stats_btn.setStyleSheet("""
            QPushButton {
                background-color: #FF9800;
                color: white;
                border: none;
                border-radius: 5px;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #F57C00;
            }
        """)
        self.reset_stats_btn.clicked.connect(self.reset_stats_clicked.emit)
        layout.addWidget(self.reset_stats_btn)
        
        group.setLayout(layout)
        return group
    
    def create_detection_settings(self) -> QGroupBox:
        """検出設定の作成"""
        group = QGroupBox("検出設定")
        layout = QVBoxLayout()
        
        # モデル選択
        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("モデル:"))
        self.model_combo = QComboBox()
        self.model_combo.addItems([
            "yolo11n.pt (最速)",
            "yolo11s.pt (バランス)",
            "yolo11m.pt (高精度)",
            "yolo11l.pt (より高精度)",
            "yolo11x.pt (最高精度)"
        ])
        self.model_combo.currentTextChanged.connect(self.on_model_changed)
        model_layout.addWidget(self.model_combo)
        layout.addLayout(model_layout)
        
        # 信頼度閾値スライダー
        confidence_layout = QVBoxLayout()
        
        # ラベル
        confidence_label_layout = QHBoxLayout()
        confidence_label_layout.addWidget(QLabel("信頼度閾値:"))
        self.confidence_value_label = QLabel("0.50")
        self.confidence_value_label.setStyleSheet("font-weight: bold;")
        confidence_label_layout.addWidget(self.confidence_value_label)
        confidence_label_layout.addStretch()
        confidence_layout.addLayout(confidence_label_layout)
        
        # スライダー
        self.confidence_slider = QSlider(Qt.Horizontal)
        self.confidence_slider.setMinimum(10)
        self.confidence_slider.setMaximum(95)
        self.confidence_slider.setValue(50)
        self.confidence_slider.setTickPosition(QSlider.TicksBelow)
        self.confidence_slider.setTickInterval(10)
        self.confidence_slider.valueChanged.connect(self.on_confidence_changed)
        confidence_layout.addWidget(self.confidence_slider)
        
        layout.addLayout(confidence_layout)
        
        # 中心点表示チェックボックス
        self.center_display_check = QCheckBox("検出ボックスの中心点を表示")
        self.center_display_check.toggled.connect(self.center_display_toggled.emit)
        layout.addWidget(self.center_display_check)
        
        group.setLayout(layout)
        return group
    
    def create_face_detection_settings(self) -> QGroupBox:
        """顔検出設定の作成"""
        group = QGroupBox("顔検出・年齢性別推定")
        group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #FF6B6B;
                border-radius: 5px;
                margin-top: 10px;
            }
            QGroupBox::title {
                color: #FF6B6B;
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """)
        
        layout = QVBoxLayout()
        
        # 顔検出チェックボックス
        self.face_detection_check = QCheckBox("顔検出を有効にする")
        self.face_detection_check.setStyleSheet("""
            QCheckBox {
                font-size: 13px;
                color: #ffffff;
            }
            QCheckBox::indicator:checked {
                background-color: #4CAF50;
            }
        """)
        self.face_detection_check.toggled.connect(self.on_face_detection_toggled)
        layout.addWidget(self.face_detection_check)
        
        # 年齢性別推定チェックボックス
        self.age_gender_check = QCheckBox("年齢・性別推定を有効にする")
        self.age_gender_check.setStyleSheet("""
            QCheckBox {
                font-size: 13px;
                color: #ffffff;
            }
            QCheckBox::indicator:checked {
                background-color: #2196F3;
            }
        """)
        self.age_gender_check.setEnabled(False)  # 顔検出が有効な時のみ使用可
        self.age_gender_check.toggled.connect(self.age_gender_toggled.emit)
        layout.addWidget(self.age_gender_check)
        
        # 顔検出信頼度スライダー
        face_conf_layout = QVBoxLayout()
        
        # ラベル
        face_conf_label_layout = QHBoxLayout()
        face_conf_label_layout.addWidget(QLabel("顔検出信頼度:"))
        self.face_confidence_value_label = QLabel("0.80")
        self.face_confidence_value_label.setStyleSheet("font-weight: bold; color: #FF6B6B;")
        face_conf_label_layout.addWidget(self.face_confidence_value_label)
        face_conf_label_layout.addStretch()
        face_conf_layout.addLayout(face_conf_label_layout)
        
        # スライダー
        self.face_confidence_slider = QSlider(Qt.Horizontal)
        self.face_confidence_slider.setMinimum(50)
        self.face_confidence_slider.setMaximum(95)
        self.face_confidence_slider.setValue(80)
        self.face_confidence_slider.setEnabled(False)  # 顔検出が有効な時のみ使用可
        self.face_confidence_slider.valueChanged.connect(self.on_face_confidence_changed)
        face_conf_layout.addWidget(self.face_confidence_slider)
        
        layout.addLayout(face_conf_layout)
        
        # 顔検出統計
        self.face_stats_layout = QGridLayout()
        self.face_stats_layout.addWidget(QLabel("検出顔数:"), 0, 0)
        self.face_count_label = QLabel("0")
        self.face_count_label.setStyleSheet("color: #FF6B6B; font-weight: bold;")
        self.face_stats_layout.addWidget(self.face_count_label, 0, 1)
        
        self.face_stats_layout.addWidget(QLabel("性別:"), 1, 0)
        self.gender_label = QLabel("M:0 F:0")
        self.gender_label.setStyleSheet("color: #9C27B0;")
        self.face_stats_layout.addWidget(self.gender_label, 1, 1)
        
        layout.addLayout(self.face_stats_layout)
        
        group.setLayout(layout)
        return group
    
    def create_camera_settings(self) -> QGroupBox:
        """カメラ設定の作成"""
        group = QGroupBox("カメラ設定")
        layout = QGridLayout()
        
        # カメラ選択（プルダウン）
        layout.addWidget(QLabel("カメラ:"), 0, 0)
        self.camera_combo = QComboBox()
        self.camera_combo.setToolTip("利用可能なカメラデバイスを選択")
        # 初期アイテムを追加
        self.camera_combo.addItem("カメラを検索中...")
        layout.addWidget(self.camera_combo, 0, 1)
        
        # カメラリフレッシュボタン
        self.refresh_camera_btn = QPushButton("🔄")
        self.refresh_camera_btn.setMaximumWidth(30)
        self.refresh_camera_btn.setToolTip("カメラリストを更新")
        self.refresh_camera_btn.clicked.connect(self.refresh_camera_list)
        layout.addWidget(self.refresh_camera_btn, 0, 2)
        
        # 解像度
        layout.addWidget(QLabel("解像度:"), 1, 0)
        self.resolution_combo = QComboBox()
        self.resolution_combo.addItems([
            "1920x1080",
            "1280x720",
            "960x540",
            "640x480"
        ])
        self.resolution_combo.setCurrentText("1280x720")
        layout.addWidget(self.resolution_combo, 1, 1)
        
        # FPS
        layout.addWidget(QLabel("FPS:"), 2, 0)
        self.fps_spin = QSpinBox()
        self.fps_spin.setMinimum(10)
        self.fps_spin.setMaximum(60)
        self.fps_spin.setValue(30)
        self.fps_spin.setSuffix(" fps")
        layout.addWidget(self.fps_spin, 2, 1)
        
        # 適用ボタン
        self.apply_camera_btn = QPushButton("適用")
        self.apply_camera_btn.clicked.connect(self.on_camera_settings_apply)
        layout.addWidget(self.apply_camera_btn, 3, 0, 1, 3)  # カラム数を3に変更
        
        group.setLayout(layout)
        return group
    
    def create_statistics_display(self) -> QGroupBox:
        """統計情報表示の作成"""
        group = QGroupBox("統計情報")
        layout = QGridLayout()
        
        # フォント設定
        stats_font = QFont()
        stats_font.setFamily("Consolas")
        stats_font.setPointSize(10)
        
        # FPS
        layout.addWidget(QLabel("FPS:"), 0, 0)
        self.fps_label = QLabel("0.0")
        self.fps_label.setFont(stats_font)
        self.fps_label.setStyleSheet("color: #00ff00;")
        layout.addWidget(self.fps_label, 0, 1)
        
        # 処理時間
        layout.addWidget(QLabel("処理時間:"), 1, 0)
        self.processing_time_label = QLabel("0.0 ms")
        self.processing_time_label.setFont(stats_font)
        self.processing_time_label.setStyleSheet("color: #00ffff;")
        layout.addWidget(self.processing_time_label, 1, 1)
        
        # 検出人数
        layout.addWidget(QLabel("検出人数:"), 2, 0)
        self.detection_count_label = QLabel("0")
        self.detection_count_label.setFont(stats_font)
        self.detection_count_label.setStyleSheet("color: #ffff00;")
        layout.addWidget(self.detection_count_label, 2, 1)
        
        # 総フレーム数
        layout.addWidget(QLabel("総フレーム:"), 3, 0)
        self.total_frames_label = QLabel("0")
        self.total_frames_label.setFont(stats_font)
        layout.addWidget(self.total_frames_label, 3, 1)
        
        # 総検出数
        layout.addWidget(QLabel("総検出数:"), 4, 0)
        self.total_detections_label = QLabel("0")
        self.total_detections_label.setFont(stats_font)
        layout.addWidget(self.total_detections_label, 4, 1)
        
        group.setLayout(layout)
        return group
    
    def on_play_pause_clicked(self):
        """再生/一時停止ボタンのクリック処理"""
        self.is_playing = not self.is_playing
        if self.is_playing:
            self.play_pause_btn.setText("⏸ 一時停止")
        else:
            self.play_pause_btn.setText("▶ 再生")
        self.play_pause_clicked.emit()
    
    def on_confidence_changed(self, value):
        """信頼度スライダーの変更処理"""
        confidence = value / 100.0
        self.confidence_value_label.setText(f"{confidence:.2f}")
        self.confidence_changed.emit(confidence)
    
    def on_model_changed(self, text):
        """モデル選択の変更処理"""
        model_name = text.split()[0]
        self.model_changed.emit(model_name)
    
    def on_camera_settings_apply(self):
        """カメラ設定の適用"""
        # 選択されたカメラのインデックスを取得
        camera_index = 0
        current_index = self.camera_combo.currentIndex()
        if current_index >= 0 and current_index < len(self.available_cameras):
            camera_index = self.available_cameras[current_index]['index']
        
        resolution = self.resolution_combo.currentText().split('x')
        settings = {
            'camera_index': camera_index,
            'width': int(resolution[0]),
            'height': int(resolution[1]),
            'fps': self.fps_spin.value()
        }
        self.camera_settings_changed.emit(settings)
        
        # ユーザーフィードバック
        camera_name = self.camera_combo.currentText()
        logger.info(f"カメラ設定を適用: {camera_name} ({resolution[0]}x{resolution[1]} @ {self.fps_spin.value()}fps)")
    
    def update_statistics(self, stats: dict):
        """統計情報の更新"""
        self.fps_label.setText(f"{stats.get('fps', 0):.1f}")
        self.processing_time_label.setText(
            f"{stats.get('processing_time', 0) * 1000:.1f} ms"
        )
        self.detection_count_label.setText(
            str(stats.get('person_count', 0))  # 'detection_count' -> 'person_count'
        )
        self.total_frames_label.setText(
            str(stats.get('frame_count', 0))  # 'total_frames' -> 'frame_count'
        )
        self.total_detections_label.setText(
            str(stats.get('total_detections', 0))
        )
        
        # 顔検出統計の更新
        if 'face_count' in stats:
            self.update_face_statistics(
                stats['face_count'],
                stats.get('gender_distribution')
            )
    
    def set_play_state(self, is_playing: bool):
        """Set play/pause state programmatically"""
        self.is_playing = is_playing
        if self.is_playing:
            self.play_pause_btn.setText("⏸ Pause")
        else:
            self.play_pause_btn.setText("▶ Play")
    
    def on_face_detection_toggled(self, checked):
        """顔検出チェックボックスの変更処理"""
        # 関連コントロールの有効/無効切り替え
        self.age_gender_check.setEnabled(checked)
        self.face_confidence_slider.setEnabled(checked)
        
        # 顔検出が無効の場合、年齢性別推定も無効にする
        if not checked:
            self.age_gender_check.setChecked(False)
        
        # シグナル送信
        self.face_detection_toggled.emit(checked)
    
    def on_face_confidence_changed(self, value):
        """顔検出信頼度スライダーの変更処理"""
        confidence = value / 100.0
        self.face_confidence_value_label.setText(f"{confidence:.2f}")
        self.face_confidence_changed.emit(confidence)
    
    def update_face_statistics(self, face_count: int, gender_dist: dict = None):
        """顔検出統計の更新"""
        self.face_count_label.setText(str(face_count))
        
        if gender_dist:
            male = gender_dist.get('Male', 0)
            female = gender_dist.get('Female', 0)
            self.gender_label.setText(f"M:{male} F:{female}")
        else:
            self.gender_label.setText("M:0 F:0")
    
    def quick_camera_init(self):
        """起動時の高速カメラ初期化（デフォルトカメラのみチェック）"""
        try:
            # デフォルトカメラのみを追加
            self.camera_combo.clear()
            self.camera_combo.addItem("デフォルトカメラ (起動中...)")
            self.available_cameras = [{
                'index': 0,
                'name': 'デフォルトカメラ',
                'available': True,
                'resolution': 'N/A',
                'fps': 30
            }]
            
            logger.info("起動時の高速カメラ初期化完了")
            
            # 後で完全なカメラリストを取得
            QTimer.singleShot(3000, self.refresh_camera_list)
            
        except Exception as e:
            logger.error(f"カメラ初期化エラー: {e}")
    
    def refresh_camera_list(self):
        """利用可能なカメラリストを更新"""
        try:
            # 現在の選択を保存
            current_camera_index = 0
            if self.camera_combo.count() > 0 and self.camera_combo.currentIndex() >= 0:
                current_index = self.camera_combo.currentIndex()
                if current_index < len(self.available_cameras):
                    current_camera_index = self.available_cameras[current_index]['index']
            
            # カメラ検出中の表示
            self.camera_combo.clear()
            self.camera_combo.addItem("カメラを検索中...")
            self.camera_combo.setEnabled(False)
            self.refresh_camera_btn.setEnabled(False)
            
            # UIを更新
            from PySide6.QtWidgets import QApplication
            QApplication.processEvents()
            
            # カメラを検出（起動後は最大3台のみチェック）
            self.available_cameras = get_available_cameras(max_test_index=3)
            
            # コンボボックスを更新
            self.camera_combo.clear()
            
            if self.available_cameras:
                # 利用可能なカメラを追加
                for camera in self.available_cameras:
                    if camera['available']:
                        # 解像度とFPS情報を含む詳細な表示
                        display_text = f"{camera['name']} ({camera['resolution']} @ {camera['fps']}fps)"
                    else:
                        display_text = f"{camera['name']} (利用不可)"
                    self.camera_combo.addItem(display_text)
                
                # 以前の選択を復元するか、最初のカメラを選択
                restored = False
                for i, camera in enumerate(self.available_cameras):
                    if camera['index'] == current_camera_index:
                        self.camera_combo.setCurrentIndex(i)
                        restored = True
                        break
                
                if not restored:
                    self.camera_combo.setCurrentIndex(0)
                
                logger.info(f"{len(self.available_cameras)}台のカメラを検出しました")
            else:
                # カメラが見つからない場合
                self.camera_combo.addItem("カメラが見つかりません")
                logger.warning("利用可能なカメラが見つかりませんでした")
            
            self.camera_combo.setEnabled(True)
            self.refresh_camera_btn.setEnabled(True)
            
        except Exception as e:
            logger.error(f"カメラリストの更新中にエラー: {e}")
            self.camera_combo.clear()
            self.camera_combo.addItem("エラー: カメラ検出失敗")
            self.camera_combo.setEnabled(True)
            self.refresh_camera_btn.setEnabled(True)
            
            # エラーメッセージを表示
            QMessageBox.warning(
                self,
                "カメラ検出エラー",
                f"カメラの検出中にエラーが発生しました:\n{str(e)}"
            )
    
    def get_selected_camera_index(self) -> int:
        """選択されているカメラのインデックスを取得"""
        current_index = self.camera_combo.currentIndex()
        if current_index >= 0 and current_index < len(self.available_cameras):
            return self.available_cameras[current_index]['index']
        return 0