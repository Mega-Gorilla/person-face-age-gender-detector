"""
Enhanced detection worker with face and age/gender estimation
"""

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal, QMutex, QWaitCondition
from PySide6.QtGui import QImage
import logging
from typing import Optional, Dict
import time

from src.pipelines.detection_pipeline import DetectionPipeline
from src.core.camera import CameraCapture
from src.ui.visualizer import Visualizer
from src.utils.performance import PerformanceMonitor

logger = logging.getLogger(__name__)


class EnhancedVisualizer(Visualizer):
    """Extended visualizer for face and age/gender display"""
    
    def draw_enhanced_detections(
        self,
        frame: np.ndarray,
        results: Dict,
        show_center: bool = False,
        show_confidence: bool = True,
        show_boxes: bool = True,
        show_labels: bool = True,
        show_faces: bool = True,
        show_age_gender: bool = True
    ) -> np.ndarray:
        """Draw enhanced detection results"""
        annotated = frame.copy()
        
        # Draw each person with their faces
        for person in results.get('persons', []):
            if show_boxes:
                # Draw person bbox in green
                px1, py1, px2, py2 = person['bbox']
                cv2.rectangle(annotated, (px1, py1), (px2, py2), (0, 255, 0), 2)
                
                # Person label
                if show_labels:
                    person_label = f"Person"
                    if show_confidence:
                        person_label += f" {person['confidence']:.2f}"
                    
                    # Calculate text size for background
                    (text_width, text_height), _ = cv2.getTextSize(
                        person_label, self.font, self.font_scale, self.thickness
                    )
                    
                    # Draw background rectangle
                    cv2.rectangle(
                        annotated,
                        (px1, py1 - text_height - 5),
                        (px1 + text_width, py1),
                        (0, 255, 0),
                        -1
                    )
                    
                    # Draw text
                    cv2.putText(
                        annotated, person_label, (px1, py1 - 5),
                        self.font, self.font_scale, (255, 255, 255), self.thickness
                    )
            
            # Draw center point if enabled
            if show_center and 'center' in person:
                cv2.circle(annotated, person['center'], 5, (255, 0, 0), -1)
            
            # Draw faces for this person
            if show_faces:
                for face in person.get('faces', []):
                    # Draw face bbox in blue
                    fx1, fy1, fx2, fy2 = face['bbox']
                    cv2.rectangle(annotated, (fx1, fy1), (fx2, fy2), (255, 0, 0), 2)
                    
                    # Age/Gender label
                    if show_age_gender and show_labels:
                        age = face.get('age')
                        age_range = face.get('age_range', 'Unknown')
                        gender = face.get('gender', 'Unknown')
                        
                        # Format label
                        if age and age != 'Unknown':
                            label = f"{gender}, {age}y"
                        else:
                            label = f"{gender}, {age_range}"
                        
                        # Calculate text size
                        (text_width, text_height), _ = cv2.getTextSize(
                            label, self.font, 0.5, 1
                        )
                        
                        # Draw label background
                        cv2.rectangle(
                            annotated,
                            (fx1, fy1 - text_height - 4),
                            (fx1 + text_width, fy1),
                            (255, 0, 0),
                            -1
                        )
                        
                        # Draw label text
                        cv2.putText(
                            annotated, label, (fx1, fy1 - 2),
                            self.font, 0.5, (255, 255, 255), 1
                        )
        
        return annotated


class EnhancedDetectionWorker(QThread):
    """Enhanced detection worker with face and age/gender estimation"""
    
    # Signals
    frame_ready = Signal(QImage)
    stats_updated = Signal(dict)
    error_occurred = Signal(str)
    detection_results = Signal(dict)  # New signal for detailed results
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Detection components
        self.pipeline = None
        self.camera = None
        self.visualizer = EnhancedVisualizer()
        self.performance = PerformanceMonitor()
        
        # Thread control
        self.mutex = QMutex()
        self.wait_condition = QWaitCondition()
        self.is_running = False
        self.is_paused = False
        
        # Settings
        self.config = {
            'person_model': 'yolo11n.pt',
            'person_confidence': 0.5,
            'enable_face_detection': True,
            'enable_age_gender': True,
            'face_in_person_only': True,
            'use_gpu': False
        }
        
        # Camera settings
        self.camera_index = 0
        self.resolution = (1280, 720)
        self.fps = 30
        
        # Display settings
        self.show_center = False
        self.show_confidence = True
        self.show_boxes = True
        self.show_labels = True
        self.show_faces = True
        self.show_age_gender = True
        
        # Performance tracking
        self.frame_count = 0
        self.last_stats_time = time.time()
        
    def initialize_components(self):
        """Initialize detection components"""
        try:
            # Initialize pipeline
            logger.info("Initializing enhanced detection pipeline...")
            self.pipeline = DetectionPipeline(self.config)
            
            # Initialize camera
            self.camera = CameraCapture(
                camera_index=self.camera_index,
                resolution=self.resolution,
                fps=self.fps
            )
            
            if not self.camera.open():
                raise RuntimeError("Failed to initialize camera")
            
            logger.info("Enhanced detection components initialized")
            return True
            
        except Exception as e:
            logger.error(f"Initialization error: {e}")
            self.error_occurred.emit(str(e))
            return False
    
    def run(self):
        """Main thread processing"""
        if not self.initialize_components():
            return
        
        self.is_running = True
        
        try:
            while self.is_running:
                # Check pause state
                self.mutex.lock()
                if self.is_paused:
                    self.wait_condition.wait(self.mutex)
                self.mutex.unlock()
                
                if not self.is_running:
                    break
                
                # Process frame
                self._process_frame()
                
        except Exception as e:
            logger.error(f"Processing error: {e}")
            self.error_occurred.emit(str(e))
        finally:
            self.cleanup()
    
    def _process_frame(self):
        """Process a single frame"""
        try:
            # Capture frame
            ret, frame = self.camera.read()
            if not ret:
                return
            
            # Run detection pipeline
            results = self.pipeline.process_frame(frame)
            
            # Update performance metrics
            self.performance.update()
            self.frame_count += 1
            
            # Visualize results
            annotated = self.visualizer.draw_enhanced_detections(
                frame,
                results,
                show_center=self.show_center,
                show_confidence=self.show_confidence,
                show_boxes=self.show_boxes,
                show_labels=self.show_labels,
                show_faces=self.show_faces,
                show_age_gender=self.show_age_gender
            )
            
            # Add performance overlay
            annotated = self.visualizer.draw_performance_overlay(
                annotated,
                self.performance.get_fps(),
                len(results['persons']),
                processing_time=results['processing_time'].get('total', 0)
            )
            
            # Convert to QImage and emit
            height, width, channel = annotated.shape
            bytes_per_line = 3 * width
            q_image = QImage(
                annotated.data,
                width,
                height,
                bytes_per_line,
                QImage.Format_RGB888
            ).rgbSwapped()
            
            self.frame_ready.emit(q_image)
            
            # Emit detailed results
            self.detection_results.emit(results)
            
            # Update statistics periodically
            current_time = time.time()
            if current_time - self.last_stats_time > 1.0:  # Update every second
                self._update_statistics(results)
                self.last_stats_time = current_time
                
        except Exception as e:
            logger.error(f"Frame processing error: {e}")
    
    def _update_statistics(self, results):
        """Update and emit statistics"""
        stats = {
            'fps': self.performance.get_fps(),
            'frame_count': self.frame_count,
            'person_count': results['statistics']['num_persons'],
            'face_count': results['statistics']['num_faces'],
            'avg_person_confidence': results['statistics']['avg_confidence']['person'],
            'avg_face_confidence': results['statistics']['avg_confidence']['face'],
            'avg_age_confidence': results['statistics']['avg_confidence']['age_gender'],
            'gender_distribution': results['statistics']['gender_distribution'],
            'age_distribution': results['statistics']['age_distribution'],
            'processing_time': results['processing_time']
        }
        
        self.stats_updated.emit(stats)
    
    def pause(self):
        """Pause detection"""
        self.mutex.lock()
        self.is_paused = True
        self.mutex.unlock()
        logger.info("Detection paused")
    
    def resume(self):
        """Resume detection"""
        self.mutex.lock()
        self.is_paused = False
        self.wait_condition.wakeAll()
        self.mutex.unlock()
        logger.info("Detection resumed")
    
    def stop(self):
        """Stop detection"""
        self.is_running = False
        self.resume()  # Wake up if paused
        
        if not self.wait(5000):  # Wait up to 5 seconds
            logger.warning("Worker thread did not stop gracefully")
            self.terminate()
            self.wait(500)
    
    def cleanup(self):
        """Cleanup resources"""
        if self.camera:
            self.camera.release()
        logger.info("Enhanced detection worker cleaned up")
    
    def update_confidence_threshold(self, value: float):
        """Update person detection confidence threshold"""
        if self.pipeline:
            self.config['person_confidence'] = value
            self.pipeline.update_config(person_confidence=value)
    
    def update_model(self, model_name: str):
        """Update detection model"""
        self.config['person_model'] = model_name
        # Model update would require pipeline restart
        logger.info(f"Model update to {model_name} - restart required")
    
    def toggle_center_display(self):
        """Toggle center point display"""
        self.show_center = not self.show_center
    
    def toggle_face_detection(self, enabled: bool):
        """Toggle face detection"""
        self.config['enable_face_detection'] = enabled
        if self.pipeline:
            self.pipeline.update_config(enable_face_detection=enabled)
    
    def toggle_age_gender(self, enabled: bool):
        """Toggle age/gender estimation"""
        self.config['enable_age_gender'] = enabled
        if self.pipeline:
            self.pipeline.update_config(enable_age_gender=enabled)
    
    def reset_stats(self):
        """Reset statistics"""
        self.performance.reset()
        self.frame_count = 0
        if self.pipeline:
            self.pipeline.reset()
        logger.info("Statistics reset")
    
    def capture_screenshot(self) -> Optional[np.ndarray]:
        """Capture current frame"""
        if self.camera:
            ret, frame = self.camera.read()
            if ret and self.pipeline:
                # Process frame with detections
                results = self.pipeline.process_frame(frame)
                
                # Draw annotations
                annotated = self.visualizer.draw_enhanced_detections(
                    frame,
                    results,
                    show_center=self.show_center,
                    show_confidence=self.show_confidence,
                    show_boxes=self.show_boxes,
                    show_labels=self.show_labels,
                    show_faces=self.show_faces,
                    show_age_gender=self.show_age_gender
                )
                
                return annotated
        return None