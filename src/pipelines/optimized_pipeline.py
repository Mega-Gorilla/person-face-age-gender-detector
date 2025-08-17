"""
Optimized detection pipeline with advanced models and batch processing
"""

import numpy as np
import cv2
from typing import Dict, List, Optional
import logging
import time
from collections import deque

from src.core.detector import PersonDetector
from src.core.face_detector_advanced import AdvancedFaceDetector
from src.core.age_gender_advanced import AdvancedAgeGenderEstimator

logger = logging.getLogger(__name__)


class OptimizedDetectionPipeline:
    """
    Optimized pipeline with state-of-the-art models and performance improvements
    """
    
    def __init__(self, config: Dict):
        """
        Initialize optimized pipeline
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        
        # Performance settings
        self.batch_size = config.get('batch_size', 4)
        self.frame_skip = config.get('frame_skip', 1)
        self.use_gpu = config.get('use_gpu', False)
        
        # Detection settings
        self.enable_face_detection = config.get('enable_face_detection', True)
        self.enable_age_gender = config.get('enable_age_gender', True)
        self.face_in_person_only = config.get('face_in_person_only', True)
        
        # Initialize components
        self._initialize_components()
        
        # Performance tracking
        self.frame_count = 0
        self.performance_stats = {
            'person_detection_time': 0,
            'face_detection_time': 0,
            'age_gender_time': 0,
            'total_time': 0
        }
        
        # Batch processing buffers
        self.face_batch = []
        self.face_batch_info = []
        
        # Temporal smoothing for stability
        self.history_window = 5
        self.detection_history = deque(maxlen=self.history_window)
        self.age_gender_history = {}  # Track ID -> history
        
        logger.info("Optimized pipeline initialized with advanced models")
    
    def _initialize_components(self):
        """Initialize detection components"""
        # Person detector (YOLOv11)
        self.person_detector = PersonDetector(
            model_name=self.config.get('person_model', 'yolo11n.pt'),
            confidence_threshold=self.config.get('person_confidence', 0.5),
            device='cuda' if self.use_gpu else 'cpu'
        )
        
        # Advanced face detector (YuNet)
        self.face_detector = AdvancedFaceDetector(
            confidence_threshold=self.config.get('face_confidence', 0.7),
            nms_threshold=0.3,
            input_size=(320, 320)
        )
        
        # Advanced age/gender estimator (ONNX models)
        self.age_gender_estimator = AdvancedAgeGenderEstimator(
            use_gpu=self.use_gpu
        )
        
        logger.info(f"Components initialized - GPU: {self.use_gpu}")
    
    def process_frame(self, frame: np.ndarray) -> Dict:
        """
        Process a single frame with optimizations
        
        Args:
            frame: Input frame
            
        Returns:
            Detection results with persons, faces, and age/gender
        """
        self.frame_count += 1
        start_time = time.time()
        
        results = {
            'persons': [],
            'faces': [],
            'statistics': {},
            'performance': {}
        }
        
        # Skip processing based on frame skip
        if self.frame_count % self.frame_skip != 0:
            return self._get_interpolated_results()
        
        # Stage 1: Person detection
        t0 = time.time()
        persons = self.person_detector.detect(frame)
        self.performance_stats['person_detection_time'] = time.time() - t0
        
        # Process each person
        for person in persons:
            person_result = {
                'bbox': person['bbox'],
                'confidence': person['confidence'],
                'center': person.get('center'),
                'faces': []
            }
            
            # Stage 2: Face detection (if enabled)
            if self.enable_face_detection:
                t1 = time.time()
                
                if self.face_in_person_only:
                    # Detect faces within person bbox
                    faces = self.face_detector.detect(frame, person['bbox'])
                else:
                    # Detect faces in full frame (once)
                    if not results['faces']:
                        faces = self.face_detector.detect(frame)
                        results['faces'] = faces
                    else:
                        faces = []
                
                self.performance_stats['face_detection_time'] += time.time() - t1
                
                # Stage 3: Age/gender estimation (if enabled)
                if self.enable_age_gender and faces:
                    t2 = time.time()
                    
                    # Collect face images for batch processing
                    face_images = []
                    for face in faces:
                        x1, y1, x2, y2 = face['bbox']
                        face_roi = frame[y1:y2, x1:x2]
                        
                        if face_roi.size > 0:
                            face_images.append(face_roi)
                            self.face_batch_info.append({
                                'person_idx': len(results['persons']),
                                'face': face
                            })
                    
                    # Batch process when buffer is full or last person
                    self.face_batch.extend(face_images)
                    
                    if len(self.face_batch) >= self.batch_size or \
                       person == persons[-1]:
                        # Process batch
                        age_gender_results = self.age_gender_estimator.batch_estimate(
                            self.face_batch
                        )
                        
                        # Assign results
                        for i, ag_result in enumerate(age_gender_results):
                            if i < len(self.face_batch_info):
                                info = self.face_batch_info[i]
                                face_data = info['face'].copy()
                                face_data.update(ag_result)
                                
                                # Apply temporal smoothing
                                face_data = self._smooth_age_gender(face_data)
                                
                                person_result['faces'].append(face_data)
                        
                        # Clear batch
                        self.face_batch = []
                        self.face_batch_info = []
                    
                    self.performance_stats['age_gender_time'] = time.time() - t2
                else:
                    # Just add face detections without age/gender
                    for face in faces:
                        person_result['faces'].append(face)
            
            results['persons'].append(person_result)
        
        # Calculate statistics
        results['statistics'] = self._calculate_statistics(results)
        
        # Performance metrics
        total_time = time.time() - start_time
        self.performance_stats['total_time'] = total_time
        results['performance'] = self.performance_stats.copy()
        
        # Add to history for interpolation
        self.detection_history.append(results)
        
        return results
    
    def _get_interpolated_results(self) -> Dict:
        """Get interpolated results for skipped frames"""
        if not self.detection_history:
            return {
                'persons': [],
                'faces': [],
                'statistics': {},
                'performance': {}
            }
        
        # Return last known results (simple interpolation)
        return self.detection_history[-1].copy()
    
    def _smooth_age_gender(self, face_data: Dict) -> Dict:
        """Apply temporal smoothing to age/gender estimates"""
        # Create unique ID for tracking (based on bbox center)
        bbox = face_data['bbox']
        center = ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)
        track_id = f"{center[0]}_{center[1]}"
        
        # Initialize history if needed
        if track_id not in self.age_gender_history:
            self.age_gender_history[track_id] = {
                'ages': deque(maxlen=self.history_window),
                'genders': deque(maxlen=self.history_window)
            }
        
        # Add to history
        history = self.age_gender_history[track_id]
        
        if 'age' in face_data:
            history['ages'].append(face_data['age'])
        
        if 'gender' in face_data:
            history['genders'].append(face_data['gender'])
        
        # Apply smoothing
        if len(history['ages']) > 1:
            # Median age for stability
            face_data['age'] = int(np.median(list(history['ages'])))
            face_data['age_smoothed'] = True
        
        if len(history['genders']) > 1:
            # Majority vote for gender
            from collections import Counter
            gender_counts = Counter(history['genders'])
            face_data['gender'] = gender_counts.most_common(1)[0][0]
            face_data['gender_smoothed'] = True
        
        # Clean old tracks
        if len(self.age_gender_history) > 100:
            # Keep only recent tracks
            self.age_gender_history = dict(
                list(self.age_gender_history.items())[-50:]
            )
        
        return face_data
    
    def _calculate_statistics(self, results: Dict) -> Dict:
        """Calculate detection statistics"""
        stats = {
            'num_persons': len(results['persons']),
            'num_faces': sum(len(p['faces']) for p in results['persons']),
            'avg_confidence': {},
            'gender_distribution': {'Male': 0, 'Female': 0, 'Unknown': 0},
            'age_distribution': {},
            'detection_methods': {}
        }
        
        # Person confidence
        if results['persons']:
            person_confs = [p['confidence'] for p in results['persons']]
            stats['avg_confidence']['person'] = np.mean(person_confs)
        
        # Face and age/gender statistics
        all_faces = []
        for person in results['persons']:
            all_faces.extend(person['faces'])
        
        if all_faces:
            # Face confidence
            face_confs = [f['confidence'] for f in all_faces if 'confidence' in f]
            if face_confs:
                stats['avg_confidence']['face'] = np.mean(face_confs)
            
            # Gender distribution
            for face in all_faces:
                gender = face.get('gender', 'Unknown')
                stats['gender_distribution'][gender] = \
                    stats['gender_distribution'].get(gender, 0) + 1
            
            # Age distribution
            for face in all_faces:
                if 'age_range' in face:
                    age_range = face['age_range']
                    stats['age_distribution'][age_range] = \
                        stats['age_distribution'].get(age_range, 0) + 1
            
            # Detection methods
            for face in all_faces:
                method = face.get('method', 'unknown')
                stats['detection_methods'][method] = \
                    stats['detection_methods'].get(method, 0) + 1
        
        # FPS calculation
        if self.performance_stats['total_time'] > 0:
            stats['fps'] = 1.0 / self.performance_stats['total_time']
        else:
            stats['fps'] = 0
        
        return stats
    
    def update_config(self, **kwargs):
        """Update pipeline configuration dynamically"""
        self.config.update(kwargs)
        
        # Update component settings
        if 'person_confidence' in kwargs:
            self.person_detector.update_threshold(kwargs['person_confidence'])
        
        if 'face_confidence' in kwargs:
            self.face_detector.update_threshold(kwargs['face_confidence'])
        
        if 'enable_face_detection' in kwargs:
            self.enable_face_detection = kwargs['enable_face_detection']
        
        if 'enable_age_gender' in kwargs:
            self.enable_age_gender = kwargs['enable_age_gender']
        
        logger.info(f"Pipeline configuration updated: {kwargs}")
    
    def reset(self):
        """Reset pipeline state"""
        self.frame_count = 0
        self.detection_history.clear()
        self.age_gender_history.clear()
        self.face_batch = []
        self.face_batch_info = []
        
        logger.info("Pipeline state reset")