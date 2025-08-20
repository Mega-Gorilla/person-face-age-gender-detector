"""
Age and Gender estimation using Caffe models with OpenCV DNN
Based on the models by Tal Hassner and Gil Levi
"""

import numpy as np
import cv2
from typing import Dict, Optional, Tuple, List
import logging
import os
from pathlib import Path
import urllib.request

logger = logging.getLogger(__name__)


class CaffeAgeGenderEstimator:
    """Age and gender estimation using proven Caffe models"""
    
    # Model mean values (standard for these Caffe models)
    MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
    
    # Age and gender lists
    AGE_LIST = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
    GENDER_LIST = ['Male', 'Female']
    
    # Model paths
    MODEL_DIR = Path("models/age_gender_caffe")
    
    # Model files
    AGE_PROTOTXT = MODEL_DIR / "age_deploy.prototxt"
    AGE_CAFFEMODEL = MODEL_DIR / "age_net.caffemodel"
    GENDER_PROTOTXT = MODEL_DIR / "gender_deploy.prototxt"
    GENDER_CAFFEMODEL = MODEL_DIR / "gender_net.caffemodel"
    
    # Model URLs (from GilLevi's original repository)
    MODEL_URLS = {
        'age_prototxt': 'https://raw.githubusercontent.com/GilLevi/AgeGenderDeepLearning/master/age_net_definitions/deploy.prototxt',
        'gender_prototxt': 'https://raw.githubusercontent.com/GilLevi/AgeGenderDeepLearning/master/gender_net_definitions/deploy.prototxt',
        'age_caffemodel': '1kiusFljZc9QfcIYdU2s7xrtWHTraHwmW',  # Google Drive ID
        'gender_caffemodel': '1W_moLzMlGiELyPxWiYQJ9KFaXroQ_NFQ'  # Google Drive ID
    }
    
    def __init__(self, use_gpu: bool = False):
        """
        Initialize age and gender estimator
        
        Args:
            use_gpu: Whether to use GPU for inference
        """
        self.use_gpu = use_gpu
        self.age_net = None
        self.gender_net = None
        
        self._initialize_models()
    
    def _download_prototxt(self, url: str, path: Path) -> bool:
        """Download prototxt file from URL"""
        try:
            logger.info(f"Downloading prototxt from {url}...")
            urllib.request.urlretrieve(url, str(path))
            logger.info(f"Downloaded to {path}")
            return True
        except Exception as e:
            logger.error(f"Failed to download prototxt: {e}")
            return False
    
    def _download_caffemodel_from_drive(self, file_id: str, path: Path) -> bool:
        """Download caffemodel from Google Drive"""
        try:
            import gdown
        except ImportError:
            logger.error("gdown is not installed. Please install it with: pip install gdown")
            logger.error("Or manually download the model files from:")
            logger.error(f"  Age model: https://drive.google.com/uc?id={self.MODEL_URLS['age_caffemodel']}")
            logger.error(f"  Gender model: https://drive.google.com/uc?id={self.MODEL_URLS['gender_caffemodel']}")
            logger.error(f"  And place them in: {self.MODEL_DIR}")
            return False
        
        try:
            logger.info(f"Downloading caffemodel from Google Drive (ID: {file_id})...")
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, str(path), quiet=False)
            logger.info(f"Downloaded to {path}")
            return True
        except Exception as e:
            logger.error(f"Failed to download caffemodel: {e}")
            logger.error(f"Please manually download from: {url}")
            logger.error(f"And save to: {path}")
            return False
    
    def _ensure_models_exist(self) -> bool:
        """Ensure all model files exist, downloading if necessary"""
        # Create model directory
        self.MODEL_DIR.mkdir(parents=True, exist_ok=True)
        
        # Check and download age prototxt
        if not self.AGE_PROTOTXT.exists():
            if not self._download_prototxt(self.MODEL_URLS['age_prototxt'], self.AGE_PROTOTXT):
                return False
        
        # Check and download gender prototxt
        if not self.GENDER_PROTOTXT.exists():
            if not self._download_prototxt(self.MODEL_URLS['gender_prototxt'], self.GENDER_PROTOTXT):
                return False
        
        # Check and download age caffemodel
        if not self.AGE_CAFFEMODEL.exists():
            logger.warning(f"Age model not found at {self.AGE_CAFFEMODEL}")
            if not self._download_caffemodel_from_drive(self.MODEL_URLS['age_caffemodel'], self.AGE_CAFFEMODEL):
                logger.error("\n" + "="*60)
                logger.error("MODEL DOWNLOAD REQUIRED")
                logger.error("="*60)
                logger.error("The age estimation model is not available.")
                logger.error("Please either:")
                logger.error("1. Install gdown: pip install gdown")
                logger.error("2. Or manually download the models and place them in:")
                logger.error(f"   {self.MODEL_DIR}/")
                logger.error("="*60 + "\n")
                return False
        
        # Check and download gender caffemodel
        if not self.GENDER_CAFFEMODEL.exists():
            logger.warning(f"Gender model not found at {self.GENDER_CAFFEMODEL}")
            if not self._download_caffemodel_from_drive(self.MODEL_URLS['gender_caffemodel'], self.GENDER_CAFFEMODEL):
                logger.error("\n" + "="*60)
                logger.error("MODEL DOWNLOAD REQUIRED")
                logger.error("="*60)
                logger.error("The gender estimation model is not available.")
                logger.error("Please either:")
                logger.error("1. Install gdown: pip install gdown")
                logger.error("2. Or manually download the models and place them in:")
                logger.error(f"   {self.MODEL_DIR}/")
                logger.error("="*60 + "\n")
                return False
        
        return True
    
    def _initialize_models(self):
        """Initialize the age/gender estimation models"""
        try:
            # Ensure models exist
            if not self._ensure_models_exist():
                logger.error("Failed to download required models")
                self._initialize_fallback()
                return
            
            # Load age model
            logger.info(f"Loading age model from {self.AGE_PROTOTXT} and {self.AGE_CAFFEMODEL}")
            self.age_net = cv2.dnn.readNet(str(self.AGE_CAFFEMODEL), str(self.AGE_PROTOTXT))
            
            # Load gender model
            logger.info(f"Loading gender model from {self.GENDER_PROTOTXT} and {self.GENDER_CAFFEMODEL}")
            self.gender_net = cv2.dnn.readNet(str(self.GENDER_CAFFEMODEL), str(self.GENDER_PROTOTXT))
            
            # Set backend and target
            if self.use_gpu and cv2.cuda.getCudaEnabledDeviceCount() > 0:
                self.age_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                self.age_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
                self.gender_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                self.gender_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
                logger.info("Using GPU acceleration for age/gender estimation")
            else:
                self.age_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                self.age_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                self.gender_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                self.gender_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                logger.info("Using CPU for age/gender estimation")
            
            self.method = "caffe"
            logger.info("Successfully initialized Caffe models for age/gender estimation")
            
        except Exception as e:
            logger.error(f"Failed to initialize Caffe models: {e}")
            self._initialize_fallback()
    
    def _initialize_fallback(self):
        """Initialize fallback method"""
        self.method = "caffe_unavailable"
        logger.error("\n" + "="*60)
        logger.error("CAFFE MODELS NOT AVAILABLE")
        logger.error("="*60)
        logger.error("Age/Gender estimation requires Caffe models.")
        logger.error("To enable age/gender estimation:")
        logger.error("1. Install gdown: pip install gdown")
        logger.error("2. Restart the application")
        logger.error("3. Models will download automatically")
        logger.error("="*60 + "\n")
    
    def estimate(
        self,
        face_image: np.ndarray,
        face_bbox: Optional[Tuple[int, int, int, int]] = None,
        person_image: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Estimate age and gender from face image
        
        Args:
            face_image: Face ROI image
            face_bbox: Face bounding box in original image
            person_image: Optional full person image for context
            
        Returns:
            Dictionary with age, age_range, gender, and confidence scores
        """
        if self.method == "caffe" and self.age_net and self.gender_net:
            return self._estimate_caffe(face_image)
        else:
            return self._estimate_fallback(face_image)
    
    def _estimate_caffe(self, face_image: np.ndarray) -> Dict:
        """Estimate using Caffe models"""
        try:
            # Create blob from face image
            blob = cv2.dnn.blobFromImage(
                face_image,
                scalefactor=1.0,
                size=(227, 227),
                mean=self.MODEL_MEAN_VALUES,
                swapRB=False,
                crop=False
            )
            
            # Gender prediction
            self.gender_net.setInput(blob)
            gender_preds = self.gender_net.forward()
            gender_idx = gender_preds[0].argmax()
            gender = self.GENDER_LIST[gender_idx]
            gender_confidence = float(gender_preds[0][gender_idx])
            
            # Age prediction
            self.age_net.setInput(blob)
            age_preds = self.age_net.forward()
            age_idx = age_preds[0].argmax()
            age_range = self.AGE_LIST[age_idx]
            age_confidence = float(age_preds[0][age_idx])
            
            # Improved age estimation with better range mapping
            # Use center values for each range
            age_mapping = {
                '(0-2)': 1,
                '(4-6)': 5,
                '(8-12)': 10,
                '(15-20)': 17.5,
                '(25-32)': 28.5,
                '(38-43)': 40.5,
                '(48-53)': 50.5,
                '(60-100)': 65  # Reduced from 70 for more realistic average
            }
            
            # Get probabilities for all age ranges
            age_probs = age_preds[0]
            
            # Method 1: Weighted average of all predictions (not just top 2)
            # This provides smoother transitions
            if len(age_probs) == len(self.AGE_LIST):
                # Calculate weighted average using all predictions
                weighted_age = 0
                total_weight = 0
                
                # Use softmax to emphasize higher probabilities
                exp_probs = np.exp(age_probs * 2)  # Temperature scaling
                softmax_probs = exp_probs / exp_probs.sum()
                
                for idx, prob in enumerate(softmax_probs):
                    age_val = age_mapping.get(self.AGE_LIST[idx], 30)
                    weighted_age += age_val * prob
                    total_weight += prob
                
                estimated_age = int(weighted_age / total_weight) if total_weight > 0 else 30
                
                # Apply confidence-based smoothing
                # If confidence is low, blend with a default age
                if age_confidence < 0.3:
                    estimated_age = int(estimated_age * 0.7 + 30 * 0.3)
            else:
                # Fallback to simple argmax
                estimated_age = age_mapping.get(age_range, 30)
            
            return {
                'age': int(estimated_age),
                'age_range': age_range,
                'age_confidence': age_confidence,
                'gender': gender,
                'gender_confidence': gender_confidence,
                'method': 'caffe'
            }
            
        except Exception as e:
            logger.error(f"Caffe estimation failed: {e}")
            return self._estimate_fallback(face_image)
    
    def _estimate_fallback(self, face_image: np.ndarray) -> Dict:
        """Return unavailable status when models not loaded"""
        return {
            'age': None,
            'age_range': 'Model Not Available',
            'age_confidence': 0.0,
            'gender': 'Model Not Available',
            'gender_confidence': 0.0,
            'method': 'caffe_unavailable'
        }
    
    def batch_estimate(self, face_images: List[np.ndarray]) -> List[Dict]:
        """Batch process multiple faces"""
        results = []
        for face_img in face_images:
            results.append(self.estimate(face_img))
        return results


def check_gdown_installed():
    """Check if gdown is installed for Google Drive downloads"""
    try:
        import gdown
        return True
    except ImportError:
        return False


def get_model_download_instructions() -> str:
    """Get instructions for downloading models"""
    return """
    ============================================================
    AGE/GENDER MODEL SETUP REQUIRED
    ============================================================
    
    The Caffe models for age and gender estimation are not installed.
    
    OPTION 1: Automatic Download (Recommended)
    ------------------------------------------
    1. Install gdown package:
       pip install gdown
       
    2. Restart the application
    
    3. Models will download automatically (~90MB total)
    
    OPTION 2: Manual Download
    -------------------------
    1. Download the following files:
    
       Age model (45.7MB):
       https://drive.google.com/uc?id=1kiusFljZc9QfcIYdU2s7xrtWHTraHwmW
       Save as: models/age_gender_caffe/age_net.caffemodel
       
       Gender model (44.9MB):
       https://drive.google.com/uc?id=1W_moLzMlGiELyPxWiYQJ9KFaXroQ_NFQ
       Save as: models/age_gender_caffe/gender_net.caffemodel
    
    2. Create the directory if it doesn't exist:
       mkdir -p models/age_gender_caffe
    
    3. Place the downloaded files in the directory
    
    ============================================================
    """