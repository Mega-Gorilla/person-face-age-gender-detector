#!/usr/bin/env python3
"""
カメラ検出機能のテストコード
"""

import sys
import unittest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import cv2
import platform

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.camera_detector import (
    get_available_cameras,
    get_camera_name,
    test_camera,
    get_camera_info
)


class TestCameraDetector(unittest.TestCase):
    """カメラ検出機能のテストクラス"""
    
    def setUp(self):
        """テストのセットアップ"""
        self.mock_cap = Mock(spec=cv2.VideoCapture)
        
    def tearDown(self):
        """テストのクリーンアップ"""
        pass
    
    # ==================== get_camera_name のテスト ====================
    
    def test_get_camera_name_default(self):
        """デフォルトカメラ名の取得テスト"""
        # インデックス0の場合
        name = get_camera_name(0)
        self.assertEqual(name, "デフォルトカメラ")
        
        # その他のインデックスの場合
        name = get_camera_name(1)
        self.assertEqual(name, "カメラ 1")
        
        name = get_camera_name(5)
        self.assertEqual(name, "カメラ 5")
    
    @patch('platform.system')
    @patch('builtins.__import__')
    def test_get_camera_name_windows_without_wmi(self, mock_import, mock_platform):
        """Windows環境でWMIが利用できない場合のテスト"""
        mock_platform.return_value = "Windows"
        
        # WMIのインポートを失敗させる
        def import_mock(name, *args, **kwargs):
            if name == 'wmi':
                raise ImportError("WMI not available")
            return __import__(name, *args, **kwargs)
        
        mock_import.side_effect = import_mock
        
        # PowerShellも失敗させる
        with patch('subprocess.run', side_effect=Exception):
            name = get_camera_name(0)
            self.assertEqual(name, "デフォルトカメラ")
    
    @patch('platform.system')
    def test_get_camera_name_macos(self, mock_platform):
        """macOS環境でのカメラ名取得テスト"""
        mock_platform.return_value = "Darwin"
        
        name = get_camera_name(0)
        self.assertEqual(name, "FaceTime HDカメラ")
        
        name = get_camera_name(1)
        self.assertEqual(name, "外部USBカメラ")
        
        name = get_camera_name(10)
        self.assertEqual(name, "カメラ 10")
    
    @patch('platform.system')
    @patch('subprocess.run')
    def test_get_camera_name_linux(self, mock_subprocess, mock_platform):
        """Linux環境でのカメラ名取得テスト"""
        mock_platform.return_value = "Linux"
        
        # v4l2-ctlコマンドが成功する場合
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "USB Camera (usb-0000:00:14.0-1):\n\t/dev/video0\n"
        mock_subprocess.return_value = mock_result
        
        name = get_camera_name(0)
        self.assertEqual(name, "USB Camera")
    
    # ==================== test_camera のテスト ====================
    
    @patch('cv2.VideoCapture')
    def test_test_camera_success(self, mock_video_capture):
        """カメラテスト成功のケース"""
        # モックの設定
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, Mock())
        mock_video_capture.return_value = mock_cap
        
        result = test_camera(0)
        self.assertTrue(result)
        mock_cap.release.assert_called_once()
    
    @patch('cv2.VideoCapture')
    def test_test_camera_not_opened(self, mock_video_capture):
        """カメラが開けない場合のテスト"""
        mock_cap = Mock()
        mock_cap.isOpened.return_value = False
        mock_video_capture.return_value = mock_cap
        
        result = test_camera(0)
        self.assertFalse(result)
    
    @patch('cv2.VideoCapture')
    def test_test_camera_read_failed(self, mock_video_capture):
        """カメラからの読み取り失敗のテスト"""
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (False, None)
        mock_video_capture.return_value = mock_cap
        
        result = test_camera(0)
        self.assertFalse(result)
        mock_cap.release.assert_called_once()
    
    @patch('cv2.VideoCapture')
    def test_test_camera_exception(self, mock_video_capture):
        """例外が発生した場合のテスト"""
        mock_video_capture.side_effect = Exception("Camera error")
        
        result = test_camera(0)
        self.assertFalse(result)
    
    # ==================== get_camera_info のテスト ====================
    
    @patch('cv2.VideoCapture')
    def test_get_camera_info_success(self, mock_video_capture):
        """カメラ情報取得成功のテスト"""
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FRAME_WIDTH: 1920.0,
            cv2.CAP_PROP_FRAME_HEIGHT: 1080.0,
            cv2.CAP_PROP_FPS: 30.0,
            cv2.CAP_PROP_FOURCC: 1196444237.0  # MJPG
        }.get(prop, 0.0)
        mock_cap.getBackendName.return_value = "DSHOW"
        mock_video_capture.return_value = mock_cap
        
        info = get_camera_info(0)
        
        self.assertIsNotNone(info)
        self.assertEqual(info['index'], 0)
        self.assertEqual(info['width'], 1920)
        self.assertEqual(info['height'], 1080)
        self.assertEqual(info['fps'], 30)
        self.assertEqual(info['backend'], "DSHOW")
        mock_cap.release.assert_called_once()
    
    @patch('cv2.VideoCapture')
    def test_get_camera_info_not_opened(self, mock_video_capture):
        """カメラが開けない場合の情報取得テスト"""
        mock_cap = Mock()
        mock_cap.isOpened.return_value = False
        mock_video_capture.return_value = mock_cap
        
        info = get_camera_info(0)
        self.assertIsNone(info)
    
    # ==================== get_available_cameras のテスト ====================
    
    @patch('cv2.VideoCapture')
    def test_get_available_cameras_single_camera(self, mock_video_capture):
        """単一カメラが利用可能な場合のテスト"""
        # カメラ0は利用可能、カメラ1以降は利用不可
        def create_mock_cap(index, backend=None):
            mock_cap = Mock()
            if index == 0:
                mock_cap.isOpened.return_value = True
                mock_cap.read.return_value = (True, Mock())
                mock_cap.get.side_effect = lambda prop: {
                    cv2.CAP_PROP_FRAME_WIDTH: 640.0,
                    cv2.CAP_PROP_FRAME_HEIGHT: 480.0,
                    cv2.CAP_PROP_FPS: 30.0,
                }.get(prop, 0.0)
            else:
                mock_cap.isOpened.return_value = False
            return mock_cap
        
        mock_video_capture.side_effect = create_mock_cap
        
        cameras = get_available_cameras(max_test_index=3)
        
        self.assertEqual(len(cameras), 1)
        self.assertEqual(cameras[0]['index'], 0)
        self.assertTrue(cameras[0]['available'])
        self.assertEqual(cameras[0]['resolution'], "640x480")
        self.assertEqual(cameras[0]['fps'], 30)
    
    @patch('cv2.VideoCapture')
    def test_get_available_cameras_multiple_cameras(self, mock_video_capture):
        """複数カメラが利用可能な場合のテスト"""
        # カメラ0と1が利用可能
        def create_mock_cap(index, backend=None):
            mock_cap = Mock()
            if index in [0, 1]:
                mock_cap.isOpened.return_value = True
                mock_cap.read.return_value = (True, Mock())
                mock_cap.get.side_effect = lambda prop: {
                    cv2.CAP_PROP_FRAME_WIDTH: 1280.0 if index == 0 else 1920.0,
                    cv2.CAP_PROP_FRAME_HEIGHT: 720.0 if index == 0 else 1080.0,
                    cv2.CAP_PROP_FPS: 30.0 if index == 0 else 60.0,
                }.get(prop, 0.0)
            else:
                mock_cap.isOpened.return_value = False
            return mock_cap
        
        mock_video_capture.side_effect = create_mock_cap
        
        cameras = get_available_cameras(max_test_index=3)
        
        self.assertEqual(len(cameras), 2)
        
        # カメラ0の確認
        self.assertEqual(cameras[0]['index'], 0)
        self.assertEqual(cameras[0]['resolution'], "1280x720")
        self.assertEqual(cameras[0]['fps'], 30)
        
        # カメラ1の確認
        self.assertEqual(cameras[1]['index'], 1)
        self.assertEqual(cameras[1]['resolution'], "1920x1080")
        self.assertEqual(cameras[1]['fps'], 60)
    
    @patch('cv2.VideoCapture')
    def test_get_available_cameras_no_cameras(self, mock_video_capture):
        """カメラが見つからない場合のテスト"""
        # すべてのカメラが利用不可
        mock_cap = Mock()
        mock_cap.isOpened.return_value = False
        mock_video_capture.return_value = mock_cap
        
        cameras = get_available_cameras(max_test_index=3)
        
        # デフォルトカメラが追加されることを確認
        self.assertEqual(len(cameras), 1)
        self.assertEqual(cameras[0]['index'], 0)
        self.assertEqual(cameras[0]['name'], 'デフォルトカメラ')
        self.assertFalse(cameras[0]['available'])
        self.assertEqual(cameras[0]['resolution'], 'N/A')
        self.assertEqual(cameras[0]['fps'], 0)
    
    @patch('cv2.VideoCapture')
    def test_get_available_cameras_invalid_fps(self, mock_video_capture):
        """無効なFPS値の場合のテスト"""
        def create_mock_cap(index, backend=None):
            mock_cap = Mock()
            if index == 0:
                mock_cap.isOpened.return_value = True
                mock_cap.read.return_value = (True, Mock())
                mock_cap.get.side_effect = lambda prop: {
                    cv2.CAP_PROP_FRAME_WIDTH: 640.0,
                    cv2.CAP_PROP_FRAME_HEIGHT: 480.0,
                    cv2.CAP_PROP_FPS: -1.0,  # 無効なFPS
                }.get(prop, 0.0)
            else:
                mock_cap.isOpened.return_value = False
            return mock_cap
        
        mock_video_capture.side_effect = create_mock_cap
        
        cameras = get_available_cameras(max_test_index=2)
        
        # FPSがデフォルト値（30）に設定されることを確認
        self.assertEqual(cameras[0]['fps'], 30)


class TestCameraDetectorIntegration(unittest.TestCase):
    """実際のカメラを使った統合テスト（オプション）"""
    
    @unittest.skipUnless(
        platform.system() in ['Windows', 'Darwin', 'Linux'],
        "実際のカメラテストはサポートされたプラットフォームでのみ実行"
    )
    def test_real_camera_detection(self):
        """実際のカメラ検出テスト（環境依存）"""
        cameras = get_available_cameras(max_test_index=2)
        
        # 少なくとも1つのエントリがあることを確認（デフォルトカメラまたは実カメラ）
        self.assertGreater(len(cameras), 0)
        
        # 各カメラの基本情報が含まれていることを確認
        for camera in cameras:
            self.assertIn('index', camera)
            self.assertIn('name', camera)
            self.assertIn('available', camera)
            self.assertIn('resolution', camera)
            self.assertIn('fps', camera)
            
            # インデックスが非負であることを確認
            self.assertGreaterEqual(camera['index'], 0)
            
            # 名前が空でないことを確認
            self.assertTrue(camera['name'])
            
            print(f"検出されたカメラ: {camera['name']} - "
                  f"Index: {camera['index']}, "
                  f"Resolution: {camera['resolution']}, "
                  f"FPS: {camera['fps']}, "
                  f"Available: {camera['available']}")


def run_tests():
    """テストを実行する"""
    # テストスイートの作成
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # 単体テストを追加
    suite.addTests(loader.loadTestsFromTestCase(TestCameraDetector))
    
    # 統合テストを追加（オプション）
    suite.addTests(loader.loadTestsFromTestCase(TestCameraDetectorIntegration))
    
    # テスト実行
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # 結果のサマリー
    print("\n" + "="*70)
    print("テスト結果サマリー")
    print("="*70)
    print(f"実行されたテスト: {result.testsRun}")
    print(f"成功: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"失敗: {len(result.failures)}")
    print(f"エラー: {len(result.errors)}")
    
    if result.failures:
        print("\n失敗したテスト:")
        for test, traceback in result.failures:
            print(f"  - {test}")
    
    if result.errors:
        print("\nエラーが発生したテスト:")
        for test, traceback in result.errors:
            print(f"  - {test}")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)