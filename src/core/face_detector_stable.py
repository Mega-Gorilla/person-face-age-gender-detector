"""
Stable face detection module with temporal smoothing and tracking
"""

import numpy as np
import cv2
from typing import List, Dict, Optional, Tuple, Deque
from collections import deque, defaultdict
import time
import logging

logger = logging.getLogger(__name__)


class StableFaceDetector:
    """
    Stable face detector with temporal smoothing and tracking
    """
    
    def __init__(
        self,
        detection_confidence: float = 0.8,
        tracking_iou_threshold: float = 0.3,
        min_detection_frames: int = 3,
        max_lost_frames: int = 5,
        temporal_window: int = 5
    ):
        """
        Initialize stable face detector
        
        Args:
            detection_confidence: Minimum confidence for detection
            tracking_iou_threshold: IoU threshold for matching faces
            min_detection_frames: Minimum frames to confirm detection
            max_lost_frames: Maximum frames to keep lost track
            temporal_window: Window size for temporal smoothing
        """
        self.detection_confidence = detection_confidence
        self.tracking_iou_threshold = tracking_iou_threshold
        self.min_detection_frames = min_detection_frames
        self.max_lost_frames = max_lost_frames
        self.temporal_window = temporal_window
        
        # Initialize detector (OpenCV Haar Cascade with optimized params)
        self._initialize_detector()
        
        # Tracking state
        self.tracks: Dict[int, FaceTrack] = {}
        self.next_track_id = 0
        self.frame_count = 0
        
        # Temporal smoothing buffers
        self.detection_history: Deque = deque(maxlen=temporal_window)
        
    def _initialize_detector(self):
        """Initialize face detector with optimized parameters"""
        try:
            # Use frontal face cascade
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            
            # Also load profile face for better coverage
            profile_path = cv2.data.haarcascades + 'haarcascade_profileface.xml'
            self.profile_cascade = cv2.CascadeClassifier(profile_path)
            
            logger.info("Initialized Haar Cascade detectors")
        except Exception as e:
            logger.error(f"Failed to initialize detectors: {e}")
            self.face_cascade = None
            self.profile_cascade = None
    
    def detect(
        self,
        frame: np.ndarray,
        person_bbox: Optional[Tuple[int, int, int, int]] = None
    ) -> List[Dict]:
        """
        Detect and track faces with stability improvements
        
        Args:
            frame: Input image
            person_bbox: Optional person bounding box
            
        Returns:
            List of stable face detections
        """
        self.frame_count += 1
        
        if self.face_cascade is None:
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
        
        # Preprocess image for better detection
        preprocessed = self._preprocess_image(roi)
        
        # Detect faces with multiple cascade configurations
        raw_detections = self._detect_faces_multi(preprocessed)
        
        # Apply NMS to remove duplicates
        detections = self._apply_nms(raw_detections)
        
        # Convert to absolute coordinates
        if person_bbox:
            for det in detections:
                det['bbox'] = (
                    det['bbox'][0] + offset[0],
                    det['bbox'][1] + offset[1],
                    det['bbox'][2] + offset[0],
                    det['bbox'][3] + offset[1]
                )
        
        # Update tracks
        self._update_tracks(detections)
        
        # Get stable detections from tracks
        stable_detections = self._get_stable_detections()
        
        # Add to history for temporal smoothing
        self.detection_history.append(stable_detections)
        
        # Apply temporal smoothing
        smoothed_detections = self._apply_temporal_smoothing()
        
        return smoothed_detections
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for better face detection"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply histogram equalization for better contrast
        gray = cv2.equalizeHist(gray)
        
        # Apply slight Gaussian blur to reduce noise
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        
        return gray
    
    def _detect_faces_multi(self, gray_image: np.ndarray) -> List[Dict]:
        """Detect faces using multiple configurations"""
        all_detections = []
        
        # Configuration sets for different scenarios
        configs = [
            # Standard detection
            {'scaleFactor': 1.1, 'minNeighbors': 5, 'minSize': (30, 30)},
            # More sensitive (catches more but more false positives)
            {'scaleFactor': 1.05, 'minNeighbors': 3, 'minSize': (20, 20)},
            # More strict (fewer false positives)
            {'scaleFactor': 1.15, 'minNeighbors': 7, 'minSize': (40, 40)},
        ]
        
        for config in configs:
            # Frontal face detection
            faces = self.face_cascade.detectMultiScale(
                gray_image,
                scaleFactor=config['scaleFactor'],
                minNeighbors=config['minNeighbors'],
                minSize=config['minSize']
            )
            
            for (x, y, w, h) in faces:
                detection = {
                    'bbox': (x, y, x + w, y + h),
                    'confidence': 0.9 - (configs.index(config) * 0.1),  # Higher confidence for stricter configs
                    'type': 'frontal'
                }
                all_detections.append(detection)
        
        # Profile face detection (if available)
        if self.profile_cascade is not None:
            profiles = self.profile_cascade.detectMultiScale(
                gray_image,
                scaleFactor=1.1,
                minNeighbors=4,
                minSize=(30, 30)
            )
            
            for (x, y, w, h) in profiles:
                detection = {
                    'bbox': (x, y, x + w, y + h),
                    'confidence': 0.7,
                    'type': 'profile'
                }
                all_detections.append(detection)
        
        return all_detections
    
    def _apply_nms(self, detections: List[Dict], iou_threshold: float = 0.3) -> List[Dict]:
        """Apply Non-Maximum Suppression to remove duplicate detections"""
        if not detections:
            return []
        
        # Convert to numpy arrays
        boxes = np.array([d['bbox'] for d in detections])
        scores = np.array([d['confidence'] for d in detections])
        
        # Apply NMS
        indices = self._nms(boxes, scores, iou_threshold)
        
        # Return filtered detections
        return [detections[i] for i in indices]
    
    def _nms(self, boxes: np.ndarray, scores: np.ndarray, threshold: float) -> List[int]:
        """Non-Maximum Suppression implementation"""
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]
        
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)
            
            inter = w * h
            iou = inter / (areas[i] + areas[order[1:]] - inter)
            
            inds = np.where(iou <= threshold)[0]
            order = order[inds + 1]
        
        return keep
    
    def _update_tracks(self, detections: List[Dict]):
        """Update face tracks with new detections"""
        # Match detections to existing tracks
        unmatched_detections = list(range(len(detections)))
        matched_tracks = set()
        
        for det_idx, detection in enumerate(detections):
            best_iou = 0
            best_track_id = None
            
            for track_id, track in self.tracks.items():
                if track_id in matched_tracks:
                    continue
                
                iou = self._calculate_iou(detection['bbox'], track.bbox)
                if iou > best_iou and iou > self.tracking_iou_threshold:
                    best_iou = iou
                    best_track_id = track_id
            
            if best_track_id is not None:
                # Update existing track
                self.tracks[best_track_id].update(detection, self.frame_count)
                matched_tracks.add(best_track_id)
                unmatched_detections.remove(det_idx)
        
        # Create new tracks for unmatched detections
        for det_idx in unmatched_detections:
            detection = detections[det_idx]
            if detection['confidence'] >= self.detection_confidence:
                track = FaceTrack(self.next_track_id, detection, self.frame_count)
                self.tracks[self.next_track_id] = track
                self.next_track_id += 1
        
        # Update lost tracks
        tracks_to_remove = []
        for track_id, track in self.tracks.items():
            if track_id not in matched_tracks:
                track.frames_since_update += 1
                if track.frames_since_update > self.max_lost_frames:
                    tracks_to_remove.append(track_id)
        
        # Remove dead tracks
        for track_id in tracks_to_remove:
            del self.tracks[track_id]
    
    def _get_stable_detections(self) -> List[Dict]:
        """Get stable detections from confirmed tracks"""
        stable_detections = []
        
        for track in self.tracks.values():
            # Only return tracks that have been detected for minimum frames
            if track.detection_count >= self.min_detection_frames:
                detection = {
                    'bbox': track.bbox,
                    'confidence': track.confidence,
                    'track_id': track.track_id,
                    'age': track.age,
                    'stable': True
                }
                stable_detections.append(detection)
        
        return stable_detections
    
    def _apply_temporal_smoothing(self) -> List[Dict]:
        """Apply temporal smoothing to detection history"""
        if not self.detection_history:
            return []
        
        # Count occurrences of each track across temporal window
        track_occurrences = defaultdict(list)
        
        for frame_detections in self.detection_history:
            for detection in frame_detections:
                track_id = detection.get('track_id')
                if track_id is not None:
                    track_occurrences[track_id].append(detection)
        
        # Build smoothed detections
        smoothed = []
        for track_id, occurrences in track_occurrences.items():
            # Only include if detected in majority of frames
            if len(occurrences) >= self.temporal_window // 2:
                # Average bbox positions for smoothing
                avg_bbox = self._average_bboxes([d['bbox'] for d in occurrences])
                avg_confidence = np.mean([d['confidence'] for d in occurrences])
                
                smoothed_detection = {
                    'bbox': avg_bbox,
                    'confidence': avg_confidence,
                    'track_id': track_id,
                    'stable': True,
                    'temporal_count': len(occurrences)
                }
                smoothed.append(smoothed_detection)
        
        return smoothed
    
    def _average_bboxes(self, bboxes: List[Tuple]) -> Tuple:
        """Calculate average of multiple bounding boxes"""
        if not bboxes:
            return (0, 0, 0, 0)
        
        bboxes = np.array(bboxes)
        avg_bbox = np.mean(bboxes, axis=0).astype(int)
        return tuple(avg_bbox)
    
    def _calculate_iou(self, bbox1: Tuple, bbox2: Tuple) -> float:
        """Calculate Intersection over Union of two bounding boxes"""
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2
        
        # Calculate intersection
        inter_xmin = max(x1_min, x2_min)
        inter_ymin = max(y1_min, y2_min)
        inter_xmax = min(x1_max, x2_max)
        inter_ymax = min(y1_max, y2_max)
        
        if inter_xmax < inter_xmin or inter_ymax < inter_ymin:
            return 0.0
        
        inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)
        
        # Calculate union
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = area1 + area2 - inter_area
        
        if union_area == 0:
            return 0.0
        
        return inter_area / union_area


class FaceTrack:
    """Face track for temporal tracking"""
    
    def __init__(self, track_id: int, detection: Dict, frame_num: int):
        self.track_id = track_id
        self.bbox = detection['bbox']
        self.confidence = detection['confidence']
        self.first_frame = frame_num
        self.last_frame = frame_num
        self.detection_count = 1
        self.frames_since_update = 0
        self.age = 0
        
        # Smoothing with Kalman filter (simplified)
        self.bbox_history = deque(maxlen=5)
        self.bbox_history.append(self.bbox)
    
    def update(self, detection: Dict, frame_num: int):
        """Update track with new detection"""
        self.bbox_history.append(detection['bbox'])
        
        # Smooth bbox using moving average
        self.bbox = self._smooth_bbox()
        
        # Update confidence with exponential moving average
        alpha = 0.3
        self.confidence = alpha * detection['confidence'] + (1 - alpha) * self.confidence
        
        self.last_frame = frame_num
        self.detection_count += 1
        self.frames_since_update = 0
        self.age = frame_num - self.first_frame
    
    def _smooth_bbox(self) -> Tuple:
        """Smooth bounding box using history"""
        if not self.bbox_history:
            return self.bbox
        
        bboxes = np.array(list(self.bbox_history))
        
        # Use median for robustness against outliers
        smoothed = np.median(bboxes, axis=0).astype(int)
        
        return tuple(smoothed)