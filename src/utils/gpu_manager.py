"""
GPU検出と管理ユーティリティ
統一的なGPU管理とデバイス自動選択機能を提供
"""

import torch
import cv2
import logging
import platform
import subprocess
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
import psutil
import warnings

logger = logging.getLogger(__name__)


@dataclass
class GPUInfo:
    """GPU情報を格納するデータクラス"""
    device_id: int
    name: str
    memory_total: int  # MB
    memory_free: int   # MB
    memory_used: int   # MB
    utilization: float  # %
    cuda_capability: Tuple[int, int]
    is_available: bool


class GPUManager:
    """統一的なGPU管理クラス"""
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        """シングルトンパターンの実装"""
        if cls._instance is None:
            cls._instance = super(GPUManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        """GPU管理の初期化"""
        if not self._initialized:
            self.cuda_available = torch.cuda.is_available()
            self.device_count = torch.cuda.device_count() if self.cuda_available else 0
            self.selected_device = None
            self.device_info = {}
            self.opencv_gpu_enabled = False
            
            # GPU情報を収集
            self._detect_gpus()
            
            # 最適なデバイスを自動選択
            self._auto_select_device()
            
            # OpenCVのGPUサポートを確認
            self._check_opencv_gpu()
            
            self._initialized = True
            
            # 初期化ログ
            self._log_gpu_status()
    
    def _detect_gpus(self) -> None:
        """利用可能なGPUを検出"""
        self.device_info = {}
        
        if not self.cuda_available:
            logger.info("CUDA対応GPUが検出されませんでした")
            return
        
        for i in range(self.device_count):
            try:
                torch.cuda.set_device(i)
                device_props = torch.cuda.get_device_properties(i)
                
                # メモリ情報取得
                memory_total = device_props.total_memory // (1024 * 1024)  # MB
                memory_free = torch.cuda.mem_get_info(i)[0] // (1024 * 1024)  # MB
                memory_used = memory_total - memory_free
                
                # GPU使用率を取得（nvidia-smiを使用）
                utilization = self._get_gpu_utilization(i)
                
                self.device_info[i] = GPUInfo(
                    device_id=i,
                    name=device_props.name,
                    memory_total=memory_total,
                    memory_free=memory_free,
                    memory_used=memory_used,
                    utilization=utilization,
                    cuda_capability=(device_props.major, device_props.minor),
                    is_available=True
                )
                
                logger.info(f"GPU {i}: {device_props.name} "
                          f"(Memory: {memory_free}/{memory_total} MB free, "
                          f"Compute Capability: {device_props.major}.{device_props.minor})")
                
            except Exception as e:
                logger.error(f"GPU {i} の情報取得に失敗: {e}")
                self.device_info[i] = GPUInfo(
                    device_id=i,
                    name="Unknown",
                    memory_total=0,
                    memory_free=0,
                    memory_used=0,
                    utilization=0.0,
                    cuda_capability=(0, 0),
                    is_available=False
                )
    
    def _get_gpu_utilization(self, device_id: int) -> float:
        """GPU使用率を取得（nvidia-smi使用）"""
        try:
            if platform.system() == "Windows":
                cmd = ["nvidia-smi", "--query-gpu=utilization.gpu",
                      "--format=csv,noheader,nounits", f"-i={device_id}"]
            else:
                cmd = ["nvidia-smi", "--query-gpu=utilization.gpu",
                      "--format=csv,noheader,nounits", f"-i", str(device_id)]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=2)
            if result.returncode == 0:
                return float(result.stdout.strip())
        except Exception:
            pass
        
        return 0.0
    
    def _auto_select_device(self) -> None:
        """最適なGPUデバイスを自動選択"""
        if not self.cuda_available or not self.device_info:
            self.selected_device = "cpu"
            logger.info("CPUモードを使用します")
            return
        
        # メモリが最も空いているGPUを選択
        best_device = None
        max_free_memory = 0
        
        for device_id, info in self.device_info.items():
            if info.is_available and info.memory_free > max_free_memory:
                # Compute Capability 3.5以上を推奨
                if info.cuda_capability[0] >= 3 and info.cuda_capability[1] >= 5:
                    best_device = device_id
                    max_free_memory = info.memory_free
        
        if best_device is not None:
            self.selected_device = f"cuda:{best_device}"
            torch.cuda.set_device(best_device)
            logger.info(f"GPU {best_device} を自動選択しました "
                       f"({self.device_info[best_device].name}, "
                       f"{max_free_memory} MB free)")
        else:
            self.selected_device = "cuda:0"
            logger.info("デフォルトGPU (cuda:0) を使用します")
    
    def _check_opencv_gpu(self) -> None:
        """OpenCVのGPUサポートを確認"""
        try:
            # OpenCVがCUDAでビルドされているか確認
            if cv2.cuda.getCudaEnabledDeviceCount() > 0:
                self.opencv_gpu_enabled = True
                logger.info(f"OpenCV CUDA サポート: 有効 "
                          f"({cv2.cuda.getCudaEnabledDeviceCount()} devices)")
            else:
                self.opencv_gpu_enabled = False
                logger.info("OpenCV CUDA サポート: 無効")
        except:
            self.opencv_gpu_enabled = False
            logger.debug("OpenCV CUDA サポートの確認に失敗")
    
    def _log_gpu_status(self) -> None:
        """GPU状態をログ出力"""
        logger.info("="*60)
        logger.info("GPU環境サマリー:")
        logger.info(f"  CUDA利用可能: {self.cuda_available}")
        logger.info(f"  検出されたGPU数: {self.device_count}")
        logger.info(f"  選択されたデバイス: {self.selected_device}")
        logger.info(f"  OpenCV GPU: {self.opencv_gpu_enabled}")
        
        if self.device_info:
            logger.info("  GPU詳細:")
            for device_id, info in self.device_info.items():
                logger.info(f"    [{device_id}] {info.name}: "
                          f"{info.memory_free}/{info.memory_total} MB free, "
                          f"使用率 {info.utilization}%")
        logger.info("="*60)
    
    def get_device(self, prefer_gpu: bool = True) -> str:
        """
        使用するデバイスを取得
        
        Args:
            prefer_gpu: GPUを優先するか
            
        Returns:
            デバイス文字列 ('cuda:0', 'cpu', etc.)
        """
        if prefer_gpu and self.cuda_available and self.selected_device != "cpu":
            return self.selected_device
        return "cpu"
    
    def get_torch_device(self, prefer_gpu: bool = True) -> torch.device:
        """PyTorchデバイスオブジェクトを取得"""
        device_str = self.get_device(prefer_gpu)
        return torch.device(device_str)
    
    def get_optimal_batch_size(self, model_type: str = "yolo") -> int:
        """
        モデルタイプに応じた最適なバッチサイズを取得
        
        Args:
            model_type: モデルの種類 ('yolo', 'face', 'age_gender')
            
        Returns:
            推奨バッチサイズ
        """
        if not self.cuda_available or self.selected_device == "cpu":
            # CPUの場合は小さいバッチサイズ
            return 1
        
        # 選択されたGPUのメモリ量に基づいてバッチサイズを決定
        device_id = int(self.selected_device.split(":")[-1])
        if device_id in self.device_info:
            free_memory = self.device_info[device_id].memory_free
            
            # モデルタイプごとの推奨バッチサイズ
            if model_type == "yolo":
                if free_memory > 8000:  # 8GB以上
                    return 8
                elif free_memory > 4000:  # 4GB以上
                    return 4
                elif free_memory > 2000:  # 2GB以上
                    return 2
                else:
                    return 1
            elif model_type == "face":
                if free_memory > 4000:
                    return 16
                elif free_memory > 2000:
                    return 8
                else:
                    return 4
            elif model_type == "age_gender":
                if free_memory > 2000:
                    return 32
                elif free_memory > 1000:
                    return 16
                else:
                    return 8
        
        return 1
    
    def update_memory_info(self) -> None:
        """GPU メモリ情報を更新"""
        if not self.cuda_available:
            return
        
        for device_id in self.device_info:
            try:
                torch.cuda.set_device(device_id)
                memory_total = torch.cuda.get_device_properties(device_id).total_memory // (1024 * 1024)
                memory_free = torch.cuda.mem_get_info(device_id)[0] // (1024 * 1024)
                
                self.device_info[device_id].memory_free = memory_free
                self.device_info[device_id].memory_used = memory_total - memory_free
                self.device_info[device_id].utilization = self._get_gpu_utilization(device_id)
                
            except Exception as e:
                logger.debug(f"GPU {device_id} のメモリ情報更新失敗: {e}")
    
    def clear_cache(self) -> None:
        """GPUキャッシュをクリア"""
        if self.cuda_available:
            torch.cuda.empty_cache()
            logger.debug("GPUキャッシュをクリアしました")
    
    def get_system_info(self) -> Dict:
        """システム情報を取得"""
        info = {
            'platform': platform.system(),
            'cpu_count': psutil.cpu_count(),
            'cpu_percent': psutil.cpu_percent(interval=0.1),
            'memory_total': psutil.virtual_memory().total // (1024 * 1024 * 1024),  # GB
            'memory_available': psutil.virtual_memory().available // (1024 * 1024 * 1024),  # GB
            'memory_percent': psutil.virtual_memory().percent,
            'cuda_available': self.cuda_available,
            'gpu_count': self.device_count,
            'selected_device': self.selected_device,
            'opencv_gpu': self.opencv_gpu_enabled
        }
        
        # GPU情報を追加
        if self.device_info:
            info['gpus'] = []
            for device_id, gpu_info in self.device_info.items():
                info['gpus'].append({
                    'id': device_id,
                    'name': gpu_info.name,
                    'memory_total': gpu_info.memory_total,
                    'memory_free': gpu_info.memory_free,
                    'utilization': gpu_info.utilization
                })
        
        return info
    
    @classmethod
    def reset(cls) -> None:
        """シングルトンインスタンスをリセット"""
        cls._instance = None
        cls._initialized = False


# グローバルGPUマネージャーインスタンス
gpu_manager = GPUManager()