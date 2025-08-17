"""
Advanced face detection module using YuNet and OpenCV DNN
"""

import numpy as np
import cv2
from typing import List, Dict, Optional, Tuple
import logging
import os
from pathlib import Path
import urllib.request

logger = logging.getLogger(__name__)


class AdvancedFaceDetector:
    """Advanced face detector using YuNet model via OpenCV DNN"""
    
    MODEL_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
    MODEL_PATH = "models/face/yunet_face_detection.onnx"
    
    def __init__(
        self,
        confidence_threshold: float = 0.7,
        nms_threshold: float = 0.3,
        top_k: int = 5000,
        input_size: Tuple[int, int] = (320, 320)
    ):
        """
        Initialize advanced face detector
        
        Args:
            confidence_threshold: Minimum confidence for detection
            nms_threshold: NMS threshold for removing duplicates
            top_k: Keep top K detections before NMS
            input_size: Input size for the model
        """
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.top_k = top_k
        self.input_size = input_size
        self.detector = None
        
        self._initialize_detector()
    
    def _download_model(self):
        """Download YuNet model if not present"""
        model_path = Path(self.MODEL_PATH)
        
        if not model_path.exists():
            logger.info(f"Downloading YuNet face detection model...")
            model_path.parent.mkdir(parents=True, exist_ok=True)
            
            try:
                urllib.request.urlretrieve(self.MODEL_URL, self.MODEL_PATH)
                logger.info(f"Model downloaded to {self.MODEL_PATH}")
            except Exception as e:
                logger.error(f"Failed to download model: {e}")
                return False
        
        return model_path.exists()
    
    def _initialize_detector(self):
        """Initialize the YuNet face detection model"""
        try:
            # Try to download model if needed
            if not self._download_model():
                logger.warning("YuNet model not available, using Haar Cascade fallback")
                self._initialize_cascade_fallback()
                return
            
            # Initialize YuNet detector
            self.detector = cv2.FaceDetectorYN.create(
                self.MODEL_PATH,
                "",
                self.input_size,
                self.confidence_threshold,
                self.nms_threshold,
                self.top_k
            )
            
            self.detection_method = "yunet"
            logger.info("Initialized YuNet face detector (state-of-the-art)")
            
        except Exception as e:
            logger.error(f"Failed to initialize YuNet: {e}")
            self._initialize_cascade_fallback()
    
    def _initialize_cascade_fallback(self):
        """Initialize Haar Cascade as fallback"""
        try:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml'
            self.cascade_detector = cv2.CascadeClassifier(cascade_path)
            self.detection_method = "cascade"
            logger.info("Using improved Haar Cascade for face detection")
        except Exception as e:
            logger.error(f"Failed to initialize cascade: {e}")
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
        if self.detector is None and self.detection_method != "cascade":
            return []
        
        # Extract ROI if person bbox provided
        if person_bbox:
            x1, y1, x2, y2 = person_bbox
            # Add padding
            padding = 30
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(frame.shape[1], x2 + padding)
            y2 = min(frame.shape[0], y2 + padding)
            roi = frame[y1:y2, x1:x2]
            offset = (x1, y1)
        else:
            roi = frame
            offset = (0, 0)
        
        # Detect faces based on method
        if self.detection_method == "yunet":
            faces = self._detect_yunet(roi)
        else:
            faces = self._detect_cascade(roi)
        
        # Adjust coordinates if ROI was used
        if person_bbox:
            for face in faces:
                face['bbox'] = (
                    face['bbox'][0] + offset[0],
                    face['bbox'][1] + offset[1],
                    face['bbox'][2] + offset[0],
                    face['bbox'][3] + offset[1]
                )
                
                if 'landmarks' in face and face['landmarks']:
                    for i in range(len(face['landmarks'])):
                        face['landmarks'][i] = (
                            face['landmarks'][i][0] + offset[0],
                            face['landmarks'][i][1] + offset[1]
                        )
        
        return faces
    
    def _detect_yunet(self, image: np.ndarray) -> List[Dict]:
        """Detect faces using YuNet model"""
        # Set input size
        h, w = image.shape[:2]
        self.detector.setInputSize((w, h))
        
        # Detect faces
        _, faces = self.detector.detect(image)
        
        if faces is None:
            return []
        
        detections = []
        for face in faces:
            # YuNet returns: x, y, width, height, landmarks(10), score
            x, y, w, h = face[:4].astype(int)
            score = face[14]  # Last element is confidence
            
            # Extract landmarks (5 points: eyes, nose, mouth corners)
            landmarks = []
            for i in range(5):
                landmarks.append((int(face[4 + i*2]), int(face[5 + i*2])))
            
            detections.append({
                'bbox': (x, y, x + w, y + h),
                'confidence': float(score),
                'landmarks': landmarks,
                'method': 'yunet'
            })
        
        return detections
    
    def _detect_cascade(self, image: np.ndarray) -> List[Dict]:
        """Detect faces using Haar Cascade with optimizations"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply histogram equalization
        gray = cv2.equalizeHist(gray)
        
        # Detect faces with optimized parameters
        faces = self.cascade_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        detections = []
        for (x, y, w, h) in faces:
            # Calculate confidence based on size and position
            confidence = min(1.0, (w * h) / (image.shape[0] * image.shape[1]) * 10)
            confidence = max(0.5, confidence)  # Minimum confidence
            
            detections.append({
                'bbox': (x, y, x + w, y + h),
                'confidence': float(confidence),
                'landmarks': None,
                'method': 'cascade'
            })
        
        # Apply NMS
        if len(detections) > 1:
            detections = self._apply_nms(detections)
        
        return detections
    
    def _apply_nms(self, detections: List[Dict]) -> List[Dict]:
        """Apply Non-Maximum Suppression"""
        if not detections:
            return detections
        
        boxes = np.array([d['bbox'] for d in detections])
        scores = np.array([d['confidence'] for d in detections])
        
        # Convert to format expected by NMS
        boxes_xyxy = boxes.astype(np.float32)
        
        # Apply NMS
        indices = cv2.dnn.NMSBoxes(
            boxes_xyxy.tolist(),
            scores.tolist(),
            self.confidence_threshold,
            self.nms_threshold
        )
        
        if indices is not None and len(indices) > 0:
            if isinstance(indices, tuple):
                indices = indices[0]
            indices = indices.flatten()
            return [detections[i] for i in indices]
        
        return []
    
    def update_threshold(self, confidence: float):
        """Update confidence threshold"""
        self.confidence_threshold = confidence
        
        if self.detection_method == "yunet" and self.detector:
            # Recreate detector with new threshold
            self.detector = cv2.FaceDetectorYN.create(
                self.MODEL_PATH,
                "",
                self.input_size,
                self.confidence_threshold,
                self.nms_threshold,
                self.top_k
            )