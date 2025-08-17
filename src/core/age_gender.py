"""
Age and Gender estimation module
"""

import numpy as np
import cv2
from typing import List, Dict, Optional, Tuple
import logging
import os

logger = logging.getLogger(__name__)

# Try to import deep learning libraries
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    logger.warning("ONNXRuntime not available. Install with: pip install onnxruntime")


class AgeGenderEstimator:
    """Age and gender estimation using deep learning models"""
    
    # Age ranges for classification
    AGE_RANGES = [
        (0, 2, "0-2"),
        (3, 6, "3-6"),
        (7, 12, "7-12"),
        (13, 19, "13-19"),
        (20, 29, "20-29"),
        (30, 39, "30-39"),
        (40, 49, "40-49"),
        (50, 59, "50-59"),
        (60, 69, "60-69"),
        (70, 100, "70+")
    ]
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        use_gpu: bool = False
    ):
        """
        Initialize age and gender estimator
        
        Args:
            model_path: Path to ONNX model file
            use_gpu: Whether to use GPU for inference
        """
        self.model_path = model_path
        self.use_gpu = use_gpu
        self.session = None
        self.method = None
        
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the age/gender estimation model"""
        if ONNX_AVAILABLE and self.model_path and os.path.exists(self.model_path):
            try:
                # Initialize ONNX Runtime session
                providers = ['CUDAExecutionProvider'] if self.use_gpu else ['CPUExecutionProvider']
                self.session = ort.InferenceSession(self.model_path, providers=providers)
                self.method = "onnx"
                logger.info(f"Loaded ONNX model from {self.model_path}")
            except Exception as e:
                logger.error(f"Failed to load ONNX model: {e}")
                self._initialize_opencv_fallback()
        else:
            self._initialize_opencv_fallback()
    
    def _initialize_opencv_fallback(self):
        """Initialize OpenCV-based age/gender estimation"""
        try:
            # Use pre-trained OpenCV models if available
            # These would need to be downloaded separately
            self.age_net = None
            self.gender_net = None
            
            # Define model paths (these would need to be downloaded)
            age_model = "age_net.caffemodel"
            age_proto = "age_deploy.prototxt"
            gender_model = "gender_net.caffemodel"
            gender_proto = "gender_deploy.prototxt"
            
            if os.path.exists(age_model) and os.path.exists(age_proto):
                self.age_net = cv2.dnn.readNet(age_model, age_proto)
                logger.info("Loaded OpenCV age model")
            
            if os.path.exists(gender_model) and os.path.exists(gender_proto):
                self.gender_net = cv2.dnn.readNet(gender_model, gender_proto)
                logger.info("Loaded OpenCV gender model")
            
            if self.age_net or self.gender_net:
                self.method = "opencv"
            else:
                self.method = "heuristic"
                logger.info("Using heuristic age/gender estimation")
                
        except Exception as e:
            logger.error(f"Failed to initialize OpenCV models: {e}")
            self.method = "heuristic"
    
    def estimate(
        self,
        frame: np.ndarray,
        face_bbox: Tuple[int, int, int, int],
        person_bbox: Optional[Tuple[int, int, int, int]] = None
    ) -> Dict:
        """
        Estimate age and gender for a face
        
        Args:
            frame: Input image
            face_bbox: Face bounding box (x1, y1, x2, y2)
            person_bbox: Optional person bounding box for context
            
        Returns:
            Dictionary with age, age_range, gender, and confidence
        """
        try:
            # Extract face region
            x1, y1, x2, y2 = face_bbox
            face_img = frame[y1:y2, x1:x2]
            
            if face_img.size == 0:
                return self._get_default_result()
            
            # Estimate based on available method
            if self.method == "onnx":
                result = self._estimate_onnx(face_img, frame, person_bbox)
            elif self.method == "opencv":
                result = self._estimate_opencv(face_img)
            else:
                result = self._estimate_heuristic(face_img)
            
            return result
            
        except Exception as e:
            logger.error(f"Age/gender estimation error: {e}")
            return self._get_default_result()
    
    def _estimate_onnx(
        self,
        face_img: np.ndarray,
        full_frame: np.ndarray,
        person_bbox: Optional[Tuple]
    ) -> Dict:
        """Estimate using ONNX model (MiVOLO-style)"""
        try:
            # Preprocess face image
            face_input = self._preprocess_image(face_img, (224, 224))
            
            # If person bbox available, extract person image for context
            if person_bbox:
                x1, y1, x2, y2 = person_bbox
                person_img = full_frame[y1:y2, x1:x2]
                person_input = self._preprocess_image(person_img, (224, 224))
                
                # Run inference with both face and person
                inputs = {
                    'face': face_input,
                    'person': person_input
                }
            else:
                # Run inference with face only
                inputs = {'face': face_input}
            
            # Run model (placeholder - actual implementation would depend on model)
            # outputs = self.session.run(None, inputs)
            
            # For now, return placeholder results
            return self._estimate_heuristic(face_img)
            
        except Exception as e:
            logger.error(f"ONNX inference error: {e}")
            return self._get_default_result()
    
    def _estimate_opencv(self, face_img: np.ndarray) -> Dict:
        """Estimate using OpenCV DNN models"""
        result = {}
        
        try:
            # Preprocess for OpenCV models
            blob = cv2.dnn.blobFromImage(
                face_img, 1.0, (227, 227),
                (78.4263377603, 87.7689143744, 114.895847746),
                swapRB=False
            )
            
            # Gender estimation
            if self.gender_net:
                self.gender_net.setInput(blob)
                gender_preds = self.gender_net.forward()
                gender_idx = gender_preds[0].argmax()
                gender = ['Male', 'Female'][gender_idx]
                gender_conf = float(gender_preds[0][gender_idx])
                result['gender'] = gender
                result['gender_confidence'] = gender_conf
            
            # Age estimation
            if self.age_net:
                self.age_net.setInput(blob)
                age_preds = self.age_net.forward()
                # Age ranges for OpenCV model
                age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)',
                           '(25-32)', '(38-43)', '(48-53)', '(60-100)']
                age_idx = age_preds[0].argmax()
                age_range = age_list[age_idx]
                result['age_range'] = age_range
                result['age'] = self._parse_age_range(age_range)
                result['age_confidence'] = float(age_preds[0][age_idx])
            
            # If no models available, use heuristic
            if not result:
                return self._estimate_heuristic(face_img)
            
            result['confidence'] = result.get('gender_confidence', 0.5) * result.get('age_confidence', 0.5)
            return result
            
        except Exception as e:
            logger.error(f"OpenCV estimation error: {e}")
            return self._estimate_heuristic(face_img)
    
    def _estimate_heuristic(self, face_img: np.ndarray) -> Dict:
        """Simple heuristic-based estimation (fallback)"""
        try:
            # Analyze face characteristics
            height, width = face_img.shape[:2]
            
            # Simple heuristics based on face size and color distribution
            # This is a placeholder - real heuristics would be more sophisticated
            
            # Estimate based on face aspect ratio and size
            aspect_ratio = height / width if width > 0 else 1.0
            face_area = height * width
            
            # Convert to grayscale for analysis
            gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            mean_intensity = np.mean(gray)
            std_intensity = np.std(gray)
            
            # Very simple heuristic (not accurate, just for demonstration)
            if aspect_ratio > 1.2:  # Longer face
                age_estimate = 35
                gender = "Male"
            else:
                age_estimate = 28
                gender = "Female"
            
            # Add some variation based on intensity
            age_estimate += int((mean_intensity - 128) / 10)
            age_estimate = max(18, min(65, age_estimate))
            
            # Find age range
            age_range = "Unknown"
            for min_age, max_age, range_label in self.AGE_RANGES:
                if min_age <= age_estimate <= max_age:
                    age_range = range_label
                    break
            
            return {
                'age': age_estimate,
                'age_range': age_range,
                'gender': gender,
                'confidence': 0.3,  # Low confidence for heuristic
                'method': 'heuristic'
            }
            
        except Exception as e:
            logger.error(f"Heuristic estimation error: {e}")
            return self._get_default_result()
    
    def _preprocess_image(
        self,
        image: np.ndarray,
        target_size: Tuple[int, int]
    ) -> np.ndarray:
        """Preprocess image for model input"""
        # Resize
        resized = cv2.resize(image, target_size)
        
        # Normalize to [0, 1]
        normalized = resized.astype(np.float32) / 255.0
        
        # Add batch dimension
        batched = np.expand_dims(normalized, axis=0)
        
        # Transpose to NCHW format if needed
        # batched = np.transpose(batched, (0, 3, 1, 2))
        
        return batched
    
    def _parse_age_range(self, age_range_str: str) -> int:
        """Parse age range string to get approximate age"""
        # Extract numbers from string like "(25-32)"
        import re
        numbers = re.findall(r'\d+', age_range_str)
        if len(numbers) >= 2:
            return (int(numbers[0]) + int(numbers[1])) // 2
        elif len(numbers) == 1:
            return int(numbers[0])
        return 30  # Default
    
    def _get_default_result(self) -> Dict:
        """Get default result when estimation fails"""
        return {
            'age': None,
            'age_range': 'Unknown',
            'gender': 'Unknown',
            'confidence': 0.0,
            'method': 'none'
        }
    
    def estimate_batch(
        self,
        frame: np.ndarray,
        face_bboxes: List[Tuple],
        person_bboxes: Optional[List[Tuple]] = None
    ) -> List[Dict]:
        """
        Estimate age and gender for multiple faces
        
        Args:
            frame: Input image
            face_bboxes: List of face bounding boxes
            person_bboxes: Optional list of person bounding boxes
            
        Returns:
            List of estimation results
        """
        results = []
        
        if person_bboxes is None:
            person_bboxes = [None] * len(face_bboxes)
        
        for face_bbox, person_bbox in zip(face_bboxes, person_bboxes):
            result = self.estimate(frame, face_bbox, person_bbox)
            results.append(result)
        
        return results


class SimpleAgeGenderEstimator:
    """Simplified age/gender estimator for testing"""
    
    def __init__(self):
        """Initialize simple estimator"""
        self.face_count = 0
    
    def estimate(self, frame: np.ndarray, face_bbox: Tuple) -> Dict:
        """Simple estimation for testing"""
        self.face_count += 1
        
        # Generate varied results for testing
        ages = [25, 30, 35, 40, 45]
        age = ages[self.face_count % len(ages)]
        
        age_ranges = ["20-29", "30-39", "40-49"]
        age_range = age_ranges[(self.face_count // 2) % len(age_ranges)]
        
        genders = ["Male", "Female"]
        gender = genders[self.face_count % 2]
        
        return {
            'age': age,
            'age_range': age_range,
            'gender': gender,
            'confidence': 0.85 + (self.face_count % 3) * 0.05,
            'method': 'simple'
        }