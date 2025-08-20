"""
カメラ検出ユーティリティ
利用可能なカメラデバイスを検出し、情報を提供する
"""

import cv2
import platform
import logging
import os
import warnings
from typing import List, Dict, Optional

# OpenCVのエラーメッセージを抑制
os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'
cv2.setLogLevel(1)  # ERROR level only

logger = logging.getLogger(__name__)


def get_available_cameras(max_test_index: int = 10) -> List[Dict[str, any]]:
    """
    利用可能なカメラデバイスを検出する
    
    Args:
        max_test_index: テストする最大カメラインデックス
    
    Returns:
        カメラ情報のリスト [{index: int, name: str, available: bool}, ...]
    """
    cameras = []
    system = platform.system()
    
    # プラットフォームに応じたバックエンドを選択
    # Windowsではまずデフォルトを試し、失敗したらDirectShowを試す
    backends = []
    if system == "Windows":
        backends = [cv2.CAP_ANY, cv2.CAP_DSHOW]
    elif system == "Darwin":
        backends = [cv2.CAP_ANY, cv2.CAP_AVFOUNDATION]
    else:
        backends = [cv2.CAP_ANY, cv2.CAP_V4L2]
    
    for index in range(max_test_index):
        # エラーメッセージを抑制してカメラをチェック
        cap = None
        
        # Windowsの場合は直接インデックスでアクセスを試みる
        if system == "Windows":
            # まずシンプルなインデックスアクセス
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    cap = cv2.VideoCapture(index)
                    if cap.isOpened():
                        # テスト読み込み
                        ret, test_frame = cap.read()
                        if not ret:
                            cap.release()
                            cap = None
                except Exception:
                    cap = None
        else:
            # 他のOSの場合はバックエンドを試す
            for backend in backends:
                try:
                    cap = cv2.VideoCapture(index, backend)
                    if cap.isOpened():
                        break
                except:
                    continue
        
        if not cap:
            continue  # 次のインデックスへ
        
        if cap.isOpened():
            # カメラが利用可能
            ret, _ = cap.read()
            if ret:
                # カメラ名を取得（可能な場合）
                camera_name = get_camera_name(index, cap)
                
                # 解像度を取得
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                if fps <= 0 or fps > 120:  # 不正なFPS値の場合
                    fps = 30  # デフォルト値
                fps = int(fps)
                
                cameras.append({
                    'index': index,
                    'name': camera_name,
                    'available': True,
                    'resolution': f"{width}x{height}",
                    'fps': fps
                })
                
                logger.debug(f"カメラ検出: Index {index} - {camera_name} ({width}x{height} @ {fps}fps)")
            
            cap.release()
    
    # カメラが見つからない場合、デフォルトを追加
    if not cameras:
        cameras.append({
            'index': 0,
            'name': 'デフォルトカメラ',
            'available': False,
            'resolution': 'N/A',
            'fps': 0
        })
        logger.warning("利用可能なカメラが見つかりませんでした")
    
    return cameras


def get_camera_name(index: int, cap: Optional[cv2.VideoCapture] = None) -> str:
    """
    カメラの名前を取得する
    
    Args:
        index: カメラインデックス
        cap: OpenCVのVideoCaptureオブジェクト
    
    Returns:
        カメラ名
    """
    system = platform.system()
    
    # Windowsの場合
    if system == "Windows":
        # WMIを使用してデバイス名を取得する試み
        try:
            import wmi
            c = wmi.WMI()
            cameras = c.Win32_PnPEntity(ConfigManagerErrorCode=0)
            camera_list = []
            
            for camera in cameras:
                if camera.Name and ('camera' in camera.Name.lower() or 
                                  'webcam' in camera.Name.lower() or
                                  'video' in camera.Name.lower() or
                                  'usb' in camera.Name.lower() and 'video' in camera.Name.lower()):
                    camera_list.append(camera.Name)
            
            if index < len(camera_list):
                return camera_list[index]
        except ImportError:
            logger.debug("WMIモジュールが利用できません")
        except Exception as e:
            logger.debug(f"WMIでのカメラ名取得失敗: {e}")
        
        # PowerShellを使用した代替方法
        try:
            import subprocess
            result = subprocess.run(
                ['powershell', '-Command', 
                 'Get-PnpDevice -Class Camera | Select-Object -ExpandProperty FriendlyName'],
                capture_output=True,
                text=True,
                timeout=2
            )
            if result.returncode == 0 and result.stdout:
                camera_names = result.stdout.strip().split('\n')
                if index < len(camera_names):
                    return camera_names[index].strip()
        except Exception:
            pass
    
    # macOSの場合
    elif system == "Darwin":
        # AVFoundationではインデックスに基づいた名前を返す
        default_names = [
            "FaceTime HDカメラ",
            "外部USBカメラ",
            "外部カメラ 2"
        ]
        if index < len(default_names):
            return default_names[index]
    
    # Linuxの場合
    elif system == "Linux":
        import subprocess
        try:
            # v4l2-ctlコマンドを使用
            result = subprocess.run(
                ['v4l2-ctl', '--list-devices'],
                capture_output=True,
                text=True,
                timeout=2
            )
            if result.returncode == 0:
                # 出力からデバイス名を解析
                lines = result.stdout.strip().split('\n')
                device_count = 0
                for i, line in enumerate(lines):
                    if not line.startswith('\t') and line.strip():
                        if device_count == index:
                            return line.split('(')[0].strip()
                        device_count += 1
        except Exception as e:
            logger.debug(f"v4l2-ctlでのカメラ名取得失敗: {e}")
    
    # デフォルト名を返す
    if index == 0:
        return "デフォルトカメラ"
    else:
        return f"カメラ {index}"


def test_camera(index: int) -> bool:
    """
    指定したインデックスのカメラが利用可能かテストする
    
    Args:
        index: カメラインデックス
    
    Returns:
        利用可能な場合True
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cap = cv2.VideoCapture(index)
            if cap.isOpened():
                ret, _ = cap.read()
                cap.release()
                return ret
        return False
    except Exception as e:
        logger.debug(f"カメラ {index} のテスト中にエラー: {e}")
        return False


def get_camera_info(index: int) -> Optional[Dict[str, any]]:
    """
    指定したカメラの詳細情報を取得する
    
    Args:
        index: カメラインデックス
    
    Returns:
        カメラ情報の辞書、取得失敗時はNone
    """
    try:
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            info = {
                'index': index,
                'name': get_camera_name(index, cap),
                'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                'fps': int(cap.get(cv2.CAP_PROP_FPS)),
                'fourcc': int(cap.get(cv2.CAP_PROP_FOURCC)),
                'backend': cap.getBackendName()
            }
            cap.release()
            return info
        return None
    except Exception as e:
        logger.error(f"カメラ {index} の情報取得中にエラー: {e}")
        return None