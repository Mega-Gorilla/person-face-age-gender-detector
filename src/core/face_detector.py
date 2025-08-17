"""
Face detection module using SCRFD/InsightFace
"""

import numpy as np
import cv2
from typing import List, Dict, Optional, Tuple
import logging
import warnings

logger = logging.getLogger(__name__)

# Try to import insightface, fallback to OpenCV if not available
try:
    import insightface
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
    logger.info("InsightFace is available for face detection")
except ImportError:
    INSIGHTFACE_AVAILABLE = False
    logger.warning("InsightFace not available, falling back to OpenCV Haar Cascade")
    warnings.warn("InsightFace not installed. Using OpenCV fallback. Install with: pip install insightface")


class FaceDetector:
    """Face detection class with InsightFace/SCRFD or OpenCV fallback"""
    
    def __init__(
        self,
        model_name: str = "buffalo_l",
        det_size: Tuple[int, int] = (640, 640),
        use_gpu: bool = False
    ):
        """
        Initialize face detector
        
        Args:
            model_name: InsightFace model name (buffalo_l, buffalo_s, etc.)
            det_size: Detection input size
            use_gpu: Whether to use GPU
        """
        self.model_name = model_name
        self.det_size = det_size
        self.use_gpu = use_gpu
        self.detector = None
        
        self._initialize_detector()
    
    def _initialize_detector(self):
        """Initialize the face detection model"""
        if INSIGHTFACE_AVAILABLE:
            try:
                # Use InsightFace with SCRFD
                providers = ['CUDAExecutionProvider'] if self.use_gpu else ['CPUExecutionProvider']
                self.detector = FaceAnalysis(
                    name=self.model_name,
                    providers=providers,
                    allowed_modules=['detection']  # Only use detection module
                )
                self.detector.prepare(ctx_id=0, det_size=self.det_size)
                self.detection_method = "insightface"
                logger.info(f"Initialized InsightFace with model: {self.model_name}")
            except Exception as e:
                logger.error(f"Failed to initialize InsightFace: {e}")
                self._initialize_opencv_fallback()
        else:
            self._initialize_opencv_fallback()
    
    def _initialize_opencv_fallback(self):
        """Initialize OpenCV Haar Cascade as fallback"""
        try:
            # Use OpenCV's pre-trained Haar Cascade
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.detector = cv2.CascadeClassifier(cascade_path)
            self.detection_method = "opencv"
            logger.info("Using OpenCV Haar Cascade for face detection")
        except Exception as e:
            logger.error(f"Failed to initialize OpenCV detector: {e}")
            self.detector = None
            self.detection_method = None
    
    def detect(
        self,
        frame: np.ndarray,
        person_bbox: Optional[Tuple[int, int, int, int]] = None
    ) -> List[Dict]:
        """
        Detect faces in frame or within person bounding box
        
        Args:
            frame: Input image frame
            person_bbox: Optional person bounding box (x1, y1, x2, y2)
            
        Returns:
            List of face detections with bbox, confidence, landmarks
        """
        if self.detector is None:
            logger.error("Face detector not initialized")
            return []
        
        try:
            # Extract ROI if person bbox provided
            if person_bbox:
                x1, y1, x2, y2 = person_bbox
                # Add padding to person bbox
                padding = 20
                x1 = max(0, x1 - padding)
                y1 = max(0, y1 - padding)
                x2 = min(frame.shape[1], x2 + padding)
                y2 = min(frame.shape[0], y2 + padding)
                roi = frame[y1:y2, x1:x2]
                offset = (x1, y1)
            else:
                roi = frame
                offset = (0, 0)
            
            # Detect faces
            if self.detection_method == "insightface":
                faces = self._detect_insightface(roi)
            elif self.detection_method == "opencv":
                faces = self._detect_opencv(roi)
            else:
                return []
            
            # Adjust coordinates if ROI was used
            if person_bbox:
                for face in faces:
                    face['bbox'] = (
                        face['bbox'][0] + offset[0],
                        face['bbox'][1] + offset[1],
                        face['bbox'][2] + offset[0],
                        face['bbox'][3] + offset[1]
                    )
                    if 'landmarks' in face and face['landmarks'] is not None:
                        face['landmarks'] = face['landmarks'] + np.array(offset)
            
            return faces
            
        except Exception as e:
            logger.error(f"Face detection error: {e}")
            return []
    
    def _detect_insightface(self, image: np.ndarray) -> List[Dict]:
        """Detect faces using InsightFace"""
        faces = self.detector.get(image)
        
        detections = []
        for face in faces:
            bbox = face.bbox.astype(int)
            detection = {
                'bbox': tuple(bbox),  # (x1, y1, x2, y2)
                'confidence': float(face.det_score),
                'landmarks': face.kps if hasattr(face, 'kps') else None,
                'embedding': face.normed_embedding if hasattr(face, 'normed_embedding') else None,
                'age': face.age if hasattr(face, 'age') else None,
                'gender': 'Male' if hasattr(face, 'gender') and face.gender == 1 else 'Female' if hasattr(face, 'gender') else None
            }
            detections.append(detection)
        
        return detections
    
    def _detect_opencv(self, image: np.ndarray) -> List[Dict]:
        """Detect faces using OpenCV Haar Cascade"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        detections = []
        for (x, y, w, h) in faces:
            detection = {
                'bbox': (x, y, x + w, y + h),
                'confidence': 0.9,  # Haar Cascade doesn't provide confidence
                'landmarks': None,
                'embedding': None,
                'age': None,
                'gender': None
            }
            detections.append(detection)
        
        return detections
    
    def detect_batch(
        self,
        frames: List[np.ndarray],
        person_bboxes: Optional[List[Tuple]] = None
    ) -> List[List[Dict]]:
        """
        Detect faces in multiple frames (batch processing)
        
        Args:
            frames: List of input frames
            person_bboxes: Optional list of person bounding boxes
            
        Returns:
            List of face detections for each frame
        """
        results = []
        
        if person_bboxes is None:
            person_bboxes = [None] * len(frames)
        
        for frame, bbox in zip(frames, person_bboxes):
            faces = self.detect(frame, bbox)
            results.append(faces)
        
        return results
    
    def update_settings(self, **kwargs):
        """Update detector settings"""
        if 'det_size' in kwargs:
            self.det_size = kwargs['det_size']
        if 'use_gpu' in kwargs:
            self.use_gpu = kwargs['use_gpu']
        
        # Reinitialize if settings changed
        self._initialize_detector()


class FaceDetectorLite:
    """Lightweight face detector using only OpenCV DNN"""
    
    def __init__(self):
        """Initialize lightweight face detector"""
        self.detector = None
        self._initialize_dnn_detector()
    
    def _initialize_dnn_detector(self):
        """Initialize OpenCV DNN face detector"""
        try:
            # Use OpenCV's DNN module with pre-trained model
            prototxt = "deploy.prototxt"
            model = "res10_300x300_ssd_iter_140000.caffemodel"
            
            # Try to use built-in OpenCV face detector
            self.detector = cv2.dnn.readNetFromCaffe(prototxt, model)
            logger.info("Initialized OpenCV DNN face detector")
        except:
            # Fallback to Haar Cascade
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.detector = cv2.CascadeClassifier(cascade_path)
            self.detection_method = "haar"
            logger.info("Using Haar Cascade as fallback")
    
    def detect(self, frame: np.ndarray, confidence_threshold: float = 0.5) -> List[Dict]:
        """Simple face detection"""
        if isinstance(self.detector, cv2.CascadeClassifier):
            # Haar Cascade detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.detector.detectMultiScale(gray, 1.1, 5)
            
            detections = []
            for (x, y, w, h) in faces:
                detections.append({
                    'bbox': (x, y, x + w, y + h),
                    'confidence': 0.9
                })
            return detections
        else:
            # DNN detection (if available)
            return []