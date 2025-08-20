#!/usr/bin/env python3
"""
Debug script for age/gender estimation to identify fluctuation issues
"""

import cv2
import numpy as np
import time
import logging
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.core.age_gender_caffe import CaffeAgeGenderEstimator
from src.core.face_detector_advanced import AdvancedFaceDetector

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AgeGenderDebugger:
    """Debug class for age/gender estimation analysis"""
    
    def __init__(self):
        """Initialize debugger with models"""
        logger.info("Initializing age/gender debugger...")
        
        # Initialize models
        self.age_gender_estimator = CaffeAgeGenderEstimator(use_gpu=False)
        self.face_detector = AdvancedFaceDetector(
            confidence_threshold=0.8,
            nms_threshold=0.3,
            input_size=(320, 320)
        )
        
        # Storage for debugging
        self.age_history = []
        self.gender_history = []
        self.confidence_history = []
        self.preprocessing_stats = []
        
    def analyze_preprocessing(self, face_roi):
        """Analyze face ROI preprocessing"""
        stats = {
            'shape': face_roi.shape,
            'dtype': str(face_roi.dtype),
            'min_val': float(face_roi.min()),
            'max_val': float(face_roi.max()),
            'mean_val': float(face_roi.mean()),
            'std_val': float(face_roi.std()),
            'is_empty': face_roi.size == 0
        }
        
        # Check if image is in correct format (BGR, uint8)
        if face_roi.dtype != np.uint8:
            logger.warning(f"Face ROI dtype is {face_roi.dtype}, expected uint8")
        
        # Check if image has correct channels
        if len(face_roi.shape) == 3 and face_roi.shape[2] != 3:
            logger.warning(f"Face ROI has {face_roi.shape[2]} channels, expected 3")
        
        # Check if image is too small
        if face_roi.shape[0] < 50 or face_roi.shape[1] < 50:
            logger.warning(f"Face ROI is very small: {face_roi.shape[:2]}")
        
        return stats
    
    def debug_blob_creation(self, face_roi):
        """Debug blob creation process"""
        # Create blob with same parameters as in CaffeAgeGenderEstimator
        blob = cv2.dnn.blobFromImage(
            face_roi,
            scalefactor=1.0,
            size=(227, 227),
            mean=(78.4263377603, 87.7689143744, 114.895847746),
            swapRB=False,
            crop=False
        )
        
        logger.debug(f"Blob shape: {blob.shape}")
        logger.debug(f"Blob dtype: {blob.dtype}")
        logger.debug(f"Blob min/max: {blob.min():.2f}/{blob.max():.2f}")
        logger.debug(f"Blob mean/std: {blob.mean():.2f}/{blob.std():.2f}")
        
        return blob
    
    def run_single_frame_analysis(self, frame):
        """Analyze single frame for debugging"""
        # Detect faces
        faces = self.face_detector.detect(frame)
        
        if not faces:
            logger.info("No faces detected")
            return None
        
        results = []
        for i, face in enumerate(faces):
            logger.info(f"\n--- Analyzing face {i+1} ---")
            
            # Extract face ROI
            x1, y1, x2, y2 = face['bbox']
            face_roi = frame[y1:y2, x1:x2]
            
            # Analyze preprocessing
            prep_stats = self.analyze_preprocessing(face_roi)
            logger.debug(f"Preprocessing stats: {prep_stats}")
            
            # Debug blob creation
            blob = self.debug_blob_creation(face_roi)
            
            # Get raw predictions
            if self.age_gender_estimator.age_net and self.age_gender_estimator.gender_net:
                # Gender prediction
                self.age_gender_estimator.gender_net.setInput(blob)
                gender_preds = self.age_gender_estimator.gender_net.forward()
                logger.debug(f"Gender predictions: {gender_preds[0]}")
                logger.debug(f"Gender argmax: {gender_preds[0].argmax()}")
                
                # Age prediction
                self.age_gender_estimator.age_net.setInput(blob)
                age_preds = self.age_gender_estimator.age_net.forward()
                logger.debug(f"Age predictions: {age_preds[0]}")
                logger.debug(f"Age argmax: {age_preds[0].argmax()}")
                
                # Get top 3 age predictions
                sorted_indices = np.argsort(age_preds[0])[::-1][:3]
                logger.info("Top 3 age predictions:")
                for idx in sorted_indices:
                    age_range = self.age_gender_estimator.AGE_LIST[idx]
                    confidence = age_preds[0][idx]
                    logger.info(f"  {age_range}: {confidence:.3f}")
            
            # Estimate using the normal method
            result = self.age_gender_estimator.estimate(face_roi, face['bbox'], frame)
            logger.info(f"Final result: Age={result['age']}, Gender={result['gender']}")
            logger.info(f"Confidence: Age={result['age_confidence']:.3f}, Gender={result['gender_confidence']:.3f}")
            
            results.append(result)
        
        return results
    
    def run_continuous_analysis(self, camera_index=0, duration=30):
        """Run continuous analysis on webcam for specified duration"""
        logger.info(f"Starting continuous analysis for {duration} seconds...")
        
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            logger.error("Failed to open camera")
            return
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        start_time = time.time()
        frame_count = 0
        
        try:
            while (time.time() - start_time) < duration:
                ret, frame = cap.read()
                if not ret:
                    logger.warning("Failed to read frame")
                    continue
                
                frame_count += 1
                
                # Process every 5th frame to reduce load
                if frame_count % 5 == 0:
                    results = self.run_single_frame_analysis(frame)
                    
                    if results:
                        for result in results:
                            if result['age'] is not None:
                                self.age_history.append(result['age'])
                                self.gender_history.append(result['gender'])
                                self.confidence_history.append({
                                    'age': result['age_confidence'],
                                    'gender': result['gender_confidence']
                                })
                    
                    # Show statistics every 10 frames
                    if len(self.age_history) > 0 and frame_count % 10 == 0:
                        self.print_statistics()
                
                # Display frame
                cv2.imshow('Age/Gender Debug', frame)
                
                # Exit on 'q' press
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
        finally:
            cap.release()
            cv2.destroyAllWindows()
            
            # Final statistics
            logger.info("\n=== FINAL STATISTICS ===")
            self.print_statistics()
    
    def print_statistics(self):
        """Print current statistics"""
        if not self.age_history:
            return
        
        logger.info("\n--- Current Statistics ---")
        logger.info(f"Total samples: {len(self.age_history)}")
        
        # Age statistics
        ages = np.array(self.age_history)
        logger.info(f"Age - Mean: {ages.mean():.1f}, Std: {ages.std():.1f}")
        logger.info(f"Age - Min: {ages.min()}, Max: {ages.max()}")
        logger.info(f"Age - Recent 10: {ages[-10:].tolist()}")
        
        # Gender distribution
        from collections import Counter
        gender_counts = Counter(self.gender_history)
        logger.info(f"Gender distribution: {dict(gender_counts)}")
        
        # Confidence statistics
        if self.confidence_history:
            age_confs = [c['age'] for c in self.confidence_history]
            gender_confs = [c['gender'] for c in self.confidence_history]
            logger.info(f"Age confidence - Mean: {np.mean(age_confs):.3f}")
            logger.info(f"Gender confidence - Mean: {np.mean(gender_confs):.3f}")


def main():
    """Main debug function"""
    debugger = AgeGenderDebugger()
    
    # Check if models are loaded
    if debugger.age_gender_estimator.method == "caffe_unavailable":
        logger.error("Caffe models not available! Please install gdown and restart.")
        return
    
    logger.info("=" * 60)
    logger.info("Age/Gender Debug Analysis")
    logger.info("=" * 60)
    logger.info("This will analyze age/gender estimation for 30 seconds")
    logger.info("Please position yourself in front of the camera")
    logger.info("Press 'q' to quit early")
    logger.info("=" * 60)
    
    # Run continuous analysis without waiting for input
    debugger.run_continuous_analysis(camera_index=0, duration=5)


if __name__ == "__main__":
    main()