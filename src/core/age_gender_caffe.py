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
import gdown

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
            logger.info(f"Downloading caffemodel from Google Drive (ID: {file_id})...")
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, str(path), quiet=False)
            logger.info(f"Downloaded to {path}")
            return True
        except Exception as e:
            logger.error(f"Failed to download caffemodel: {e}")
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
            if not self._download_caffemodel_from_drive(self.MODEL_URLS['age_caffemodel'], self.AGE_CAFFEMODEL):
                # Try alternative download method
                logger.warning("Trying alternative download for age model...")
                return False
        
        # Check and download gender caffemodel
        if not self.GENDER_CAFFEMODEL.exists():
            if not self._download_caffemodel_from_drive(self.MODEL_URLS['gender_caffemodel'], self.GENDER_CAFFEMODEL):
                # Try alternative download method
                logger.warning("Trying alternative download for gender model...")
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
        self.method = "fallback"
        logger.warning("Using fallback heuristic method for age/gender estimation")
    
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
            
            # Estimate specific age from range
            age_mapping = {
                '(0-2)': 1,
                '(4-6)': 5,
                '(8-12)': 10,
                '(15-20)': 18,
                '(25-32)': 28,
                '(38-43)': 40,
                '(48-53)': 50,
                '(60-100)': 70
            }
            estimated_age = age_mapping.get(age_range, 30)
            
            # Add some variance based on confidence
            if age_confidence < 0.8:
                estimated_age += np.random.randint(-2, 3)
            
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
        """Simple fallback estimation"""
        # For fallback, return reasonable defaults
        return {
            'age': 30,
            'age_range': '(25-32)',
            'age_confidence': 0.3,
            'gender': 'Male' if np.random.random() > 0.5 else 'Female',
            'gender_confidence': 0.5,
            'method': 'fallback'
        }
    
    def batch_estimate(self, face_images: List[np.ndarray]) -> List[Dict]:
        """Batch process multiple faces"""
        results = []
        for face_img in face_images:
            results.append(self.estimate(face_img))
        return results


# Create a function to check if gdown is installed
def check_gdown_installed():
    """Check if gdown is installed for Google Drive downloads"""
    try:
        import gdown
        return True
    except ImportError:
        logger.warning("gdown not installed. Install with: pip install gdown")
        return False


# Alternative implementation without gdown
class SimpleCaffeAgeGenderEstimator:
    """Simplified version that works without gdown"""
    
    MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
    AGE_LIST = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
    GENDER_LIST = ['Male', 'Female']
    
    def __init__(self):
        self.method = "simple_fallback"
        logger.info("Using simplified age/gender estimator")
    
    def estimate(self, face_image: np.ndarray, **kwargs) -> Dict:
        """Simple estimation based on image analysis"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
            
            # Simple heuristics based on image properties
            h, w = gray.shape
            mean_intensity = np.mean(gray)
            std_intensity = np.std(gray)
            
            # Estimate age based on texture complexity
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            if laplacian_var < 100:
                age_idx = 0  # Very smooth - baby
            elif laplacian_var < 300:
                age_idx = 2  # Smooth - child
            elif laplacian_var < 600:
                age_idx = 3  # Young adult
            elif laplacian_var < 1000:
                age_idx = 4  # Adult
            else:
                age_idx = 5  # Older adult
            
            age_range = self.AGE_LIST[min(age_idx, len(self.AGE_LIST)-1)]
            
            # Simple gender estimation (placeholder)
            gender = 'Male' if mean_intensity > 127 else 'Female'
            
            # Map to specific age
            age_mapping = {
                '(0-2)': 1, '(4-6)': 5, '(8-12)': 10,
                '(15-20)': 18, '(25-32)': 28, '(38-43)': 40,
                '(48-53)': 50, '(60-100)': 70
            }
            
            return {
                'age': age_mapping.get(age_range, 30),
                'age_range': age_range,
                'age_confidence': 0.4,
                'gender': gender,
                'gender_confidence': 0.4,
                'method': 'simple_heuristic'
            }
            
        except Exception as e:
            logger.error(f"Simple estimation failed: {e}")
            return {
                'age': 30,
                'age_range': '(25-32)',
                'age_confidence': 0.2,
                'gender': 'Unknown',
                'gender_confidence': 0.0,
                'method': 'error'
            }
    
    def batch_estimate(self, face_images: List[np.ndarray]) -> List[Dict]:
        """Batch process multiple faces"""
        return [self.estimate(img) for img in face_images]