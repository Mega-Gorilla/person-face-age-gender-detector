"""
File processing worker thread for video file detection
"""

import cv2
import json
import csv
import time
from pathlib import Path
from PySide6.QtCore import QThread, Signal
import logging
from typing import Optional, Dict, List, Any
import numpy as np

from src.core.detector import PersonDetector
from src.ui.visualizer import Visualizer
from src.utils.coordinate_exporter import CoordinateExporter

logger = logging.getLogger(__name__)

class FileProcessingWorker(QThread):
    """Worker thread for processing video files"""
    
    # Signals
    progress_updated = Signal(int, int, float, float)  # current, total, fps, elapsed
    frame_processed = Signal(int, float, list)  # frame_num, timestamp, detections
    processing_completed = Signal(dict)  # output files
    error_occurred = Signal(str)
    log_message = Signal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.detector = None
        self.visualizer = Visualizer()
        self.exporter = CoordinateExporter()
        
        self.is_running = False
        self.params = {}
        
    def set_parameters(self, params: dict):
        """Set processing parameters"""
        self.params = params
        logger.info(f"Processing parameters set: {params}")
        
    def run(self):
        """Main processing loop"""
        try:
            self.is_running = True
            self.log_message.emit("Initializing detector...")
            
            # Initialize detector
            self.detector = PersonDetector(
                model_name=self.params['model'],
                confidence_threshold=self.params['confidence']
            )
            
            # Open video file
            self.log_message.emit(f"Opening video file: {self.params['input_file']}")
            cap = cv2.VideoCapture(self.params['input_file'])
            
            if not cap.isOpened():
                raise RuntimeError(f"Failed to open video file: {self.params['input_file']}")
                
            # Get video properties
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            self.log_message.emit(
                f"Video info: {width}x{height} @ {fps:.1f}fps, {total_frames} frames"
            )
            
            # Prepare output files
            output_files = self._prepare_output_files(width, height, fps)
            
            # Process frames
            self._process_video(cap, total_frames, fps, output_files)
            
            # Cleanup
            cap.release()
            self._finalize_output_files(output_files)
            
            # Emit completion
            output_paths = {
                'video': output_files.get('video_path', ''),
                'data': output_files.get('data_path', '')
            }
            self.processing_completed.emit(output_paths)
            
        except Exception as e:
            logger.error(f"Processing error: {e}")
            self.error_occurred.emit(str(e))
        finally:
            self.is_running = False
            
    def _prepare_output_files(self, width: int, height: int, fps: float) -> dict:
        """Prepare output files"""
        output_files = {}
        input_path = Path(self.params['input_file'])
        output_dir = Path(self.params['output_dir'])
        base_name = input_path.stem
        
        # Video writer
        if self.params['export_video']:
            video_format = self.params['video_format']
            video_path = output_dir / f"{base_name}_detected{video_format}"
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v') if video_format == '.mp4' else cv2.VideoWriter_fourcc(*'XVID')
            output_files['video_writer'] = cv2.VideoWriter(
                str(video_path),
                fourcc,
                fps,
                (width, height)
            )
            output_files['video_path'] = str(video_path)
            self.log_message.emit(f"Output video: {video_path}")
            
        # Data file
        if self.params['export_data']:
            data_format = self.params['data_format']
            data_path = output_dir / f"{base_name}_detections{data_format}"
            output_files['data_path'] = str(data_path)
            output_files['detections_data'] = []
            self.log_message.emit(f"Output data: {data_path}")
            
        # Frames directory
        if self.params['export_frames']:
            frames_dir = output_dir / f"{base_name}_frames"
            frames_dir.mkdir(exist_ok=True)
            output_files['frames_dir'] = frames_dir
            self.log_message.emit(f"Output frames: {frames_dir}")
            
        return output_files
        
    def _process_video(self, cap: cv2.VideoCapture, total_frames: int, 
                      fps: float, output_files: dict):
        """Process video frames"""
        frame_count = 0
        processed_count = 0
        start_time = time.time()
        frame_skip = self.params['frame_skip']
        
        while self.is_running and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            
            # Skip frames if needed
            if (frame_count - 1) % frame_skip != 0:
                continue
                
            processed_count += 1
            timestamp = frame_count / fps
            
            # Detect persons
            detections = self.detector.detect(frame)
            
            # Draw annotations if needed
            if self.params['export_video'] or self.params['export_frames']:
                annotated_frame = self.visualizer.draw_detections(
                    frame,
                    detections,
                    show_center=False,  # Don't show center point for file processing
                    show_confidence=self.params['draw_confidence'],
                    show_boxes=self.params['draw_boxes'],
                    show_labels=self.params['draw_labels']
                )
            else:
                annotated_frame = frame
                
            # Write video frame
            if self.params['export_video'] and 'video_writer' in output_files:
                output_files['video_writer'].write(annotated_frame)
                
            # Save individual frame
            if self.params['export_frames'] and 'frames_dir' in output_files:
                frame_path = output_files['frames_dir'] / f"frame_{frame_count:06d}.jpg"
                cv2.imwrite(str(frame_path), annotated_frame)
                
            # Store detection data
            if self.params['export_data']:
                frame_data = {
                    'frame': frame_count,
                    'timestamp': timestamp,
                    'detections': []
                }
                
                for detection in detections:
                    # Convert bbox tuple to list
                    bbox = detection['bbox']
                    if isinstance(bbox, tuple):
                        bbox = list(bbox)
                    elif hasattr(bbox, 'tolist'):
                        bbox = bbox.tolist()
                    
                    det_data = {
                        'bbox': bbox,
                        'confidence': float(detection['confidence']),
                        'class': detection.get('class_name', 'person')  # Default to 'person' if not specified
                    }
                    frame_data['detections'].append(det_data)
                    
                output_files['detections_data'].append(frame_data)
                
            # Calculate statistics
            elapsed = time.time() - start_time
            current_fps = processed_count / elapsed if elapsed > 0 else 0
            
            # Emit progress
            self.progress_updated.emit(frame_count, total_frames, current_fps, elapsed)
            
            # Emit frame result
            if detections:
                avg_conf = np.mean([d['confidence'] for d in detections])
                objects = f"{len(detections)} person(s)"
            else:
                avg_conf = 0.0
                objects = "No detections"
                
            self.frame_processed.emit(frame_count, timestamp, detections)
            
    def _finalize_output_files(self, output_files: dict):
        """Finalize and close output files"""
        # Close video writer
        if 'video_writer' in output_files:
            output_files['video_writer'].release()
            self.log_message.emit("Video file saved successfully")
            
        # Save detection data
        if self.params['export_data'] and 'detections_data' in output_files:
            data_path = output_files['data_path']
            data_format = self.params['data_format']
            
            if data_format == '.json':
                with open(data_path, 'w') as f:
                    json.dump(output_files['detections_data'], f, indent=2)
            elif data_format == '.csv':
                self._save_csv(data_path, output_files['detections_data'])
            elif data_format == '.xml':
                self._save_xml(data_path, output_files['detections_data'])
                
            self.log_message.emit(f"Detection data saved to {data_path}")
            
    def _save_csv(self, file_path: str, data: List[Dict]):
        """Save detection data as CSV"""
        with open(file_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Frame', 'Timestamp', 'Detection_ID', 'X1', 'Y1', 'X2', 'Y2', 'Confidence', 'Class'])
            
            for frame_data in data:
                frame = frame_data['frame']
                timestamp = frame_data['timestamp']
                
                for i, detection in enumerate(frame_data['detections']):
                    bbox = detection['bbox']
                    writer.writerow([
                        frame, timestamp, i,
                        bbox[0], bbox[1], bbox[2], bbox[3],
                        detection['confidence'],
                        detection['class']
                    ])
                    
    def _save_xml(self, file_path: str, data: List[Dict]):
        """Save detection data as XML"""
        import xml.etree.ElementTree as ET
        
        root = ET.Element('detections')
        
        for frame_data in data:
            frame_elem = ET.SubElement(root, 'frame')
            frame_elem.set('number', str(frame_data['frame']))
            frame_elem.set('timestamp', f"{frame_data['timestamp']:.3f}")
            
            for detection in frame_data['detections']:
                det_elem = ET.SubElement(frame_elem, 'detection')
                det_elem.set('class', detection['class'])
                det_elem.set('confidence', f"{detection['confidence']:.3f}")
                
                bbox_elem = ET.SubElement(det_elem, 'bbox')
                bbox = detection['bbox']
                bbox_elem.set('x1', str(bbox[0]))
                bbox_elem.set('y1', str(bbox[1]))
                bbox_elem.set('x2', str(bbox[2]))
                bbox_elem.set('y2', str(bbox[3]))
                
        tree = ET.ElementTree(root)
        tree.write(file_path, encoding='utf-8', xml_declaration=True)
        
    def stop(self):
        """Stop processing"""
        self.is_running = False
        self.log_message.emit("Stopping processing...")
        
        # Wait for thread to finish
        if not self.wait(2000):
            logger.warning("File processing thread did not stop gracefully")
            self.terminate()
            self.wait(500)