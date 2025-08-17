"""
Stable detection pipeline with improved face detection and tracking
"""

import numpy as np
import time
from typing import List, Dict, Optional, Any
import logging

from src.core.detector import PersonDetector
from src.core.face_detector_stable import StableFaceDetector
from src.core.age_gender import AgeGenderEstimator, SimpleAgeGenderEstimator

logger = logging.getLogger(__name__)

# Try to use advanced models if available
ADVANCED_MODELS_AVAILABLE = False
try:
    from src.core.face_detector_advanced import AdvancedFaceDetector
    from src.core.age_gender_advanced import AdvancedAgeGenderEstimator
    # Import Caffe models
    from src.core.age_gender_caffe import CaffeAgeGenderEstimator, check_gdown_installed, get_model_download_instructions
    ADVANCED_MODELS_AVAILABLE = True
    logger.info("Advanced models available, using optimized implementations")
except ImportError as e:
    ADVANCED_MODELS_AVAILABLE = False
    logger.info("Advanced models not available, using standard implementations")


class StableDetectionPipeline:
    """
    Improved detection pipeline with stable face detection and tracking
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize stable detection pipeline
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Feature flags
        self.enable_face_detection = self.config.get('enable_face_detection', True)
        self.enable_age_gender = self.config.get('enable_age_gender', True)
        self.enable_face_in_person_only = self.config.get('face_in_person_only', True)
        
        # Stability parameters
        self.face_detection_confidence = self.config.get('face_confidence', 0.8)
        self.face_tracking_iou = self.config.get('face_tracking_iou', 0.3)
        self.min_face_frames = self.config.get('min_face_frames', 3)
        self.temporal_window = self.config.get('temporal_window', 5)
        
        # Use advanced models preference
        self.use_advanced_models = self.config.get('use_advanced_models', True) and ADVANCED_MODELS_AVAILABLE
        
        # Initialize components
        self.person_detector = None
        self.face_detector = None
        self.age_gender_estimator = None
        
        # Age/Gender smoothing
        self.age_gender_history = {}  # track_id -> history
        
        # Frame counter
        self.frame_count = 0
        
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize detection components"""
        try:
            # Person detector
            logger.info("Initializing person detector...")
            self.person_detector = PersonDetector(
                model_name=self.config.get('person_model', 'yolo11n.pt'),
                confidence_threshold=self.config.get('person_confidence', 0.5)
            )
            
            # Face detector - use advanced if available
            if self.enable_face_detection:
                if self.use_advanced_models:
                    logger.info("Initializing advanced face detector (YuNet)...")
                    self.face_detector = AdvancedFaceDetector(
                        confidence_threshold=self.face_detection_confidence,
                        nms_threshold=0.3,
                        input_size=(320, 320)
                    )
                else:
                    logger.info("Initializing stable face detector...")
                    self.face_detector = StableFaceDetector(
                        detection_confidence=self.face_detection_confidence,
                        tracking_iou_threshold=self.face_tracking_iou,
                        min_detection_frames=self.min_face_frames,
                        temporal_window=self.temporal_window
                    )
            
            # Age/Gender estimator - use Caffe models
            if self.enable_age_gender:
                if self.use_advanced_models:
                    # Use Caffe models (most reliable)
                    try:
                        logger.info("Initializing Caffe age/gender estimator...")
                        self.age_gender_estimator = CaffeAgeGenderEstimator(
                            use_gpu=self.config.get('use_gpu', False)
                        )
                        
                        # Check if initialization was successful
                        if hasattr(self.age_gender_estimator, 'method') and self.age_gender_estimator.method == 'caffe_unavailable':
                            logger.warning("\n" + get_model_download_instructions())
                            # Still use the estimator but it will return "Model Not Available"
                            
                    except Exception as e:
                        logger.error(f"Failed to initialize Caffe models: {e}")
                        logger.error("\n" + get_model_download_instructions())
                        # Fallback to ONNX models as last resort
                        logger.info("Attempting fallback to ONNX age/gender estimator...")
                        try:
                            self.age_gender_estimator = AdvancedAgeGenderEstimator(
                                use_gpu=self.config.get('use_gpu', False)
                            )
                        except:
                            # Create a dummy estimator that returns unavailable
                            self.age_gender_estimator = None
                            logger.error("No age/gender estimation models available")
                else:
                    logger.info("Initializing standard age/gender estimator...")
                    try:
                        self.age_gender_estimator = AgeGenderEstimator(
                            model_path=self.config.get('age_gender_model'),
                            use_gpu=self.config.get('use_gpu', False)
                        )
                    except Exception as e:
                        logger.warning(f"Failed to initialize AgeGenderEstimator: {e}")
                        self.age_gender_estimator = SimpleAgeGenderEstimator()
            
            logger.info("Stable pipeline initialization complete")
            
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
        Process a single frame with stable detection
        
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
        
        # Results structure
        results = {
            'frame_number': self.frame_count,
            'persons': [],
            'faces': [],
            'statistics': {},
            'processing_time': {},
            'stability_info': {}
        }
        
        # Step 1: Person detection
        person_start = time.time()
        persons = self.person_detector.detect(frame)
        results['processing_time']['person_detection'] = time.time() - person_start
        
        # Track stability info
        results['stability_info']['person_count'] = len(persons)
        
        # Process each detected person
        for person_idx, person in enumerate(persons):
            person['id'] = f"person_{person_idx}"
            person['faces'] = []
            
            # Step 2: Stable face detection
            if do_face and self.face_detector:
                face_start = time.time()
                
                # Detect faces with tracking
                if self.enable_face_in_person_only:
                    faces = self.face_detector.detect(frame, person['bbox'])
                else:
                    if person_idx == 0:
                        faces = self.face_detector.detect(frame)
                    else:
                        faces = []
                
                # Process each face
                for face_idx, face in enumerate(faces):
                    # Add face metadata
                    face['id'] = f"face_{person_idx}_{face_idx}"
                    face['person_id'] = person['id']
                    
                    # Get track ID for stability
                    track_id = face.get('track_id')
                    
                    # Step 3: Age/Gender estimation with smoothing
                    if do_age and track_id is not None:
                        if self.age_gender_estimator:
                            age_start = time.time()
                            
                            # Extract face ROI for age/gender estimation
                            x1, y1, x2, y2 = face['bbox']
                            face_roi = frame[y1:y2, x1:x2]
                            
                            if face_roi.size > 0:
                                # Estimate age/gender
                                age_gender = self.age_gender_estimator.estimate(
                                    face_roi,
                                    face['bbox'],
                                    frame  # Pass full frame as person image
                                )
                                
                                # Apply temporal smoothing for age/gender
                                smoothed_age_gender = self._smooth_age_gender(track_id, age_gender)
                                face.update(smoothed_age_gender)
                            else:
                                # Empty ROI
                                face.update({
                                    'age': None,
                                    'age_range': 'ROI Error',
                                    'gender': 'ROI Error',
                                    'age_confidence': 0.0,
                                    'gender_confidence': 0.0,
                                    'method': 'error'
                                })
                            
                            if person_idx == 0 and face_idx == 0:
                                results['processing_time']['age_gender'] = time.time() - age_start
                        else:
                            # No estimator available
                            face.update({
                                'age': None,
                                'age_range': 'Model Not Available',
                                'gender': 'Model Not Available',
                                'age_confidence': 0.0,
                                'gender_confidence': 0.0,
                                'method': 'none'
                            })
                    
                    person['faces'].append(face)
                    results['faces'].append(face)
                
                if person_idx == 0:
                    results['processing_time']['face_detection'] = time.time() - face_start
            
            results['persons'].append(person)
        
        # Calculate statistics
        results['statistics'] = self._calculate_statistics(results)
        results['processing_time']['total'] = time.time() - start_time
        
        # Add stability metrics
        results['stability_info']['stable_faces'] = sum(
            1 for face in results['faces'] if face.get('stable', False)
        )
        results['stability_info']['tracked_faces'] = sum(
            1 for face in results['faces'] if face.get('track_id') is not None
        )
        
        return results
    
    def _smooth_age_gender(self, track_id: int, age_gender: Dict) -> Dict:
        """Apply temporal smoothing to age/gender estimates"""
        # Initialize history if needed
        if track_id not in self.age_gender_history:
            self.age_gender_history[track_id] = {
                'ages': [],
                'genders': [],
                'confidences': []
            }
        
        history = self.age_gender_history[track_id]
        
        # Add to history (keep last 10 estimates)
        if age_gender.get('age') is not None:
            history['ages'].append(age_gender['age'])
            history['ages'] = history['ages'][-10:]
        
        if age_gender.get('gender') is not None:
            history['genders'].append(age_gender['gender'])
            history['genders'] = history['genders'][-10:]
        
        if age_gender.get('confidence') is not None:
            history['confidences'].append(age_gender['confidence'])
            history['confidences'] = history['confidences'][-10:]
        
        # Calculate smoothed values
        smoothed = age_gender.copy()
        
        # Smooth age (median for robustness)
        if history['ages']:
            smoothed['age'] = int(np.median(history['ages']))
            
            # Determine age range from smoothed age
            age = smoothed['age']
            if age < 18:
                smoothed['age_range'] = "0-17"
            elif age < 30:
                smoothed['age_range'] = "18-29"
            elif age < 45:
                smoothed['age_range'] = "30-44"
            elif age < 60:
                smoothed['age_range'] = "45-59"
            else:
                smoothed['age_range'] = "60+"
        
        # Smooth gender (majority vote)
        if history['genders']:
            from collections import Counter
            gender_counts = Counter(history['genders'])
            smoothed['gender'] = gender_counts.most_common(1)[0][0]
        
        # Smooth confidence (mean)
        if history['confidences']:
            smoothed['confidence'] = np.mean(history['confidences'])
            smoothed['smoothed'] = True
        
        return smoothed
    
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
            },
            'stability_metrics': {
                'stable_ratio': 0.0,
                'tracking_ratio': 0.0
            }
        }
        
        # Person confidence
        if results['persons']:
            person_confs = [p['confidence'] for p in results['persons']]
            stats['avg_confidence']['person'] = np.mean(person_confs)
        
        # Face and age/gender statistics
        stable_count = 0
        tracked_count = 0
        
        if results['faces']:
            face_confs = []
            age_confs = []
            
            for face in results['faces']:
                if 'confidence' in face:
                    face_confs.append(face['confidence'])
                
                # Count stable and tracked faces
                if face.get('stable', False):
                    stable_count += 1
                if face.get('track_id') is not None:
                    tracked_count += 1
                
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
            
            # Stability metrics
            if results['faces']:
                stats['stability_metrics']['stable_ratio'] = stable_count / len(results['faces'])
                stats['stability_metrics']['tracking_ratio'] = tracked_count / len(results['faces'])
        
        return stats
    
    def reset(self):
        """Reset pipeline state"""
        self.frame_count = 0
        self.age_gender_history.clear()
        
        if self.face_detector:
            # Reset face detector tracking if attributes exist
            if hasattr(self.face_detector, 'tracks'):
                self.face_detector.tracks.clear()
            if hasattr(self.face_detector, 'next_track_id'):
                self.face_detector.next_track_id = 0
            if hasattr(self.face_detector, 'frame_count'):
                self.face_detector.frame_count = 0
            if hasattr(self.face_detector, 'detection_history'):
                self.face_detector.detection_history.clear()
        
        logger.info("Stable pipeline state reset")
    
    def update_config(self, **kwargs):
        """Update pipeline configuration"""
        self.config.update(kwargs)
        
        # Update feature flags
        if 'enable_face_detection' in kwargs:
            self.enable_face_detection = kwargs['enable_face_detection']
        if 'enable_age_gender' in kwargs:
            self.enable_age_gender = kwargs['enable_age_gender']
        
        # Update stability parameters
        if 'face_confidence' in kwargs and self.face_detector:
            self.face_detector.detection_confidence = kwargs['face_confidence']
        
        # Update person detector
        if 'person_confidence' in kwargs and self.person_detector:
            self.person_detector.update_threshold(kwargs['person_confidence'])
    
    def get_performance_metrics(self) -> Dict:
        """Get pipeline performance metrics"""
        face_tracks = 0
        if self.face_detector and hasattr(self.face_detector, 'tracks'):
            face_tracks = len(self.face_detector.tracks)
        
        metrics = {
            'total_frames': self.frame_count,
            'face_detector_tracks': face_tracks,
            'age_gender_cache_size': len(self.age_gender_history)
        }
        
        return metrics