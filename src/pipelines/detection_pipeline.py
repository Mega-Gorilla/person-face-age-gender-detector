"""
Integrated detection pipeline for person, face, and age/gender estimation
"""

import numpy as np
import time
from typing import List, Dict, Optional, Any
import logging

from src.core.detector import PersonDetector
from src.core.face_detector import FaceDetector, FaceDetectorLite
from src.core.age_gender import AgeGenderEstimator, SimpleAgeGenderEstimator

logger = logging.getLogger(__name__)


class DetectionPipeline:
    """
    Integrated pipeline for hierarchical detection:
    Person → Face → Age/Gender
    """
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize detection pipeline
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Feature flags
        self.enable_face_detection = self.config.get('enable_face_detection', True)
        self.enable_age_gender = self.config.get('enable_age_gender', True)
        self.enable_face_in_person_only = self.config.get('face_in_person_only', True)
        
        # Performance settings
        self.face_detection_skip_frames = self.config.get('face_skip_frames', 1)
        self.age_gender_skip_frames = self.config.get('age_skip_frames', 2)
        
        # Initialize components
        self.person_detector = None
        self.face_detector = None
        self.age_gender_estimator = None
        
        # Frame counters for skip logic
        self.frame_count = 0
        self.last_face_frame = -1
        self.last_age_frame = -1
        
        # Cache for temporal smoothing
        self.detection_cache = {}
        
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize detection components based on configuration"""
        try:
            # Person detector (always required)
            logger.info("Initializing person detector...")
            self.person_detector = PersonDetector(
                model_name=self.config.get('person_model', 'yolo11n.pt'),
                confidence_threshold=self.config.get('person_confidence', 0.5)
            )
            
            # Face detector (optional)
            if self.enable_face_detection:
                logger.info("Initializing face detector...")
                try:
                    self.face_detector = FaceDetector(
                        model_name=self.config.get('face_model', 'buffalo_l'),
                        use_gpu=self.config.get('use_gpu', False)
                    )
                except Exception as e:
                    logger.warning(f"Failed to initialize InsightFace, using lite version: {e}")
                    self.face_detector = FaceDetectorLite()
            
            # Age/Gender estimator (optional)
            if self.enable_age_gender:
                logger.info("Initializing age/gender estimator...")
                try:
                    self.age_gender_estimator = AgeGenderEstimator(
                        model_path=self.config.get('age_gender_model'),
                        use_gpu=self.config.get('use_gpu', False)
                    )
                except Exception as e:
                    logger.warning(f"Failed to initialize age/gender estimator, using simple version: {e}")
                    self.age_gender_estimator = SimpleAgeGenderEstimator()
            
            logger.info("Pipeline initialization complete")
            
        except Exception as e:
            logger.error(f"Pipeline initialization error: {e}")
            raise
    
    def process_frame(
        self,
        frame: np.ndarray,
        detect_faces: Optional[bool] = None,
        estimate_age_gender: Optional[bool] = None
    ) -> Dict:
        """
        Process a single frame through the detection pipeline
        
        Args:
            frame: Input image frame
            detect_faces: Override face detection setting
            estimate_age_gender: Override age/gender estimation setting
            
        Returns:
            Dictionary containing all detection results
        """
        start_time = time.time()
        self.frame_count += 1
        
        # Override settings if specified
        do_face = self.enable_face_detection if detect_faces is None else detect_faces
        do_age = self.enable_age_gender if estimate_age_gender is None else estimate_age_gender
        
        # Check skip frames logic
        should_detect_faces = do_face and (
            self.frame_count - self.last_face_frame >= self.face_detection_skip_frames
        )
        should_estimate_age = do_age and (
            self.frame_count - self.last_age_frame >= self.age_gender_skip_frames
        )
        
        # Results structure
        results = {
            'frame_number': self.frame_count,
            'persons': [],
            'faces': [],
            'statistics': {},
            'processing_time': {}
        }
        
        # Step 1: Person detection (always performed)
        person_start = time.time()
        persons = self.person_detector.detect(frame)
        results['processing_time']['person_detection'] = time.time() - person_start
        
        # Process each detected person
        for person_idx, person in enumerate(persons):
            person['id'] = f"person_{person_idx}"
            person['faces'] = []
            
            # Step 2: Face detection (if enabled)
            if should_detect_faces and self.face_detector:
                face_start = time.time()
                
                # Detect faces within person bbox or full frame
                if self.enable_face_in_person_only:
                    faces = self.face_detector.detect(frame, person['bbox'])
                else:
                    # Detect in full frame if this is the first person
                    if person_idx == 0:
                        all_faces = self.face_detector.detect(frame)
                        # Assign faces to persons based on overlap
                        faces = self._assign_faces_to_person(all_faces, person['bbox'])
                    else:
                        faces = []
                
                # Process each face
                for face_idx, face in enumerate(faces):
                    face['id'] = f"face_{person_idx}_{face_idx}"
                    face['person_id'] = person['id']
                    
                    # Step 3: Age/Gender estimation (if enabled)
                    if should_estimate_age and self.age_gender_estimator:
                        age_start = time.time()
                        age_gender = self.age_gender_estimator.estimate(
                            frame,
                            face['bbox'],
                            person['bbox']
                        )
                        face.update(age_gender)
                        
                        if person_idx == 0 and face_idx == 0:
                            results['processing_time']['age_gender'] = time.time() - age_start
                    
                    person['faces'].append(face)
                    results['faces'].append(face)
                
                if person_idx == 0:
                    results['processing_time']['face_detection'] = time.time() - face_start
                
                # Update last frame counters
                if should_detect_faces:
                    self.last_face_frame = self.frame_count
                if should_estimate_age:
                    self.last_age_frame = self.frame_count
            
            results['persons'].append(person)
        
        # Calculate statistics
        results['statistics'] = self._calculate_statistics(results)
        results['processing_time']['total'] = time.time() - start_time
        
        # Update cache for temporal smoothing
        self._update_cache(results)
        
        return results
    
    def _assign_faces_to_person(
        self,
        faces: List[Dict],
        person_bbox: tuple
    ) -> List[Dict]:
        """Assign faces to a person based on bbox overlap"""
        assigned_faces = []
        px1, py1, px2, py2 = person_bbox
        
        for face in faces:
            fx1, fy1, fx2, fy2 = face['bbox']
            
            # Check if face center is within person bbox
            face_center_x = (fx1 + fx2) / 2
            face_center_y = (fy1 + fy2) / 2
            
            if (px1 <= face_center_x <= px2 and py1 <= face_center_y <= py2):
                assigned_faces.append(face)
        
        return assigned_faces
    
    def _calculate_statistics(self, results: Dict) -> Dict:
        """Calculate detection statistics"""
        stats = {
            'num_persons': len(results['persons']),
            'num_faces': len(results['faces']),
            'faces_per_person': len(results['faces']) / max(1, len(results['persons'])),
            'gender_distribution': {'Male': 0, 'Female': 0, 'Unknown': 0},
            'age_distribution': {},
            'avg_confidence': {
                'person': 0.0,
                'face': 0.0,
                'age_gender': 0.0
            }
        }
        
        # Person confidence
        if results['persons']:
            person_confs = [p['confidence'] for p in results['persons']]
            stats['avg_confidence']['person'] = np.mean(person_confs)
        
        # Face and age/gender statistics
        if results['faces']:
            face_confs = []
            age_confs = []
            
            for face in results['faces']:
                if 'confidence' in face:
                    face_confs.append(face['confidence'])
                
                # Gender distribution
                gender = face.get('gender', 'Unknown')
                stats['gender_distribution'][gender] = stats['gender_distribution'].get(gender, 0) + 1
                
                # Age distribution
                age_range = face.get('age_range', 'Unknown')
                stats['age_distribution'][age_range] = stats['age_distribution'].get(age_range, 0) + 1
                
                # Age/gender confidence
                if 'confidence' in face and face.get('age') is not None:
                    age_confs.append(face['confidence'])
            
            if face_confs:
                stats['avg_confidence']['face'] = np.mean(face_confs)
            if age_confs:
                stats['avg_confidence']['age_gender'] = np.mean(age_confs)
        
        return stats
    
    def _update_cache(self, results: Dict):
        """Update detection cache for temporal smoothing"""
        # Simple cache update - can be extended for tracking
        self.detection_cache[self.frame_count] = {
            'persons': len(results['persons']),
            'faces': len(results['faces']),
            'timestamp': time.time()
        }
        
        # Keep only recent frames in cache
        max_cache_size = 30
        if len(self.detection_cache) > max_cache_size:
            oldest_frame = min(self.detection_cache.keys())
            del self.detection_cache[oldest_frame]
    
    def process_batch(
        self,
        frames: List[np.ndarray]
    ) -> List[Dict]:
        """
        Process multiple frames in batch
        
        Args:
            frames: List of input frames
            
        Returns:
            List of detection results
        """
        results = []
        for frame in frames:
            result = self.process_frame(frame)
            results.append(result)
        return results
    
    def update_config(self, **kwargs):
        """Update pipeline configuration"""
        self.config.update(kwargs)
        
        # Update feature flags
        if 'enable_face_detection' in kwargs:
            self.enable_face_detection = kwargs['enable_face_detection']
        if 'enable_age_gender' in kwargs:
            self.enable_age_gender = kwargs['enable_age_gender']
        if 'face_in_person_only' in kwargs:
            self.enable_face_in_person_only = kwargs['face_in_person_only']
        
        # Update component settings
        if 'person_confidence' in kwargs and self.person_detector:
            self.person_detector.update_threshold(kwargs['person_confidence'])
    
    def get_performance_metrics(self) -> Dict:
        """Get pipeline performance metrics"""
        if not self.detection_cache:
            return {}
        
        # Calculate average processing times
        recent_frames = list(self.detection_cache.values())[-10:]
        
        metrics = {
            'avg_persons': np.mean([f['persons'] for f in recent_frames]),
            'avg_faces': np.mean([f['faces'] for f in recent_frames]),
            'cache_size': len(self.detection_cache),
            'total_frames': self.frame_count
        }
        
        return metrics
    
    def reset(self):
        """Reset pipeline state"""
        self.frame_count = 0
        self.last_face_frame = -1
        self.last_age_frame = -1
        self.detection_cache.clear()
        logger.info("Pipeline state reset")