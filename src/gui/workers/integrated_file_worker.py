"""
統合型ファイル処理ワーカースレッド（顔検出・年齢性別推定対応）
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
from src.pipelines.stable_detection_pipeline import StableDetectionPipeline
from src.ui.visualizer import Visualizer
from src.utils.coordinate_exporter import CoordinateExporter
from src.utils.video_codec import VideoCodecManager

logger = logging.getLogger(__name__)


class IntegratedVisualizerFile(Visualizer):
    """拡張ビジュアライザー（ファイル処理用）"""
    
    def draw_integrated_detections(
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
        """統合検出結果の描画"""
        annotated = frame.copy()
        
        # パイプライン結果の場合
        if 'persons' in results:
            for person in results['persons']:
                if show_boxes:
                    # 人物バウンディングボックス描画
                    x1, y1, x2, y2 = person['bbox']
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                    if show_labels:
                        # 人物ラベル
                        if show_confidence:
                            label = f"Person {person['confidence']:.2f}"
                        else:
                            label = "Person"
                        
                        cv2.putText(annotated, label, (x1, y1 - 10),
                                   self.font, self.font_scale, (0, 255, 0), self.thickness)
                
                # 中心点表示
                if show_center and 'center' in person:
                    cv2.circle(annotated, person['center'], 5, (255, 0, 0), -1)
                
                # 顔検出結果の描画
                if show_faces and 'faces' in person:
                    for face in person['faces']:
                        fx1, fy1, fx2, fy2 = face['bbox']
                        
                        # 安定した顔は青、不安定な顔は赤
                        if face.get('stable', False):
                            face_color = (255, 0, 0)  # Blue
                            face_thickness = 2
                        else:
                            face_color = (0, 0, 255)  # Red
                            face_thickness = 1
                        
                        cv2.rectangle(annotated, (fx1, fy1), (fx2, fy2), 
                                    face_color, face_thickness)
                        
                        # 年齢・性別表示
                        if show_age_gender and show_labels:
                            age = face.get('age')
                            gender = face.get('gender', 'Unknown')
                            
                            if age:
                                ag_label = f"{gender}, {age}"
                            else:
                                ag_label = f"{gender}"
                            
                            # ラベル背景
                            label_size, _ = cv2.getTextSize(ag_label, self.font, 0.4, 1)
                            cv2.rectangle(annotated,
                                        (fx1, fy2),
                                        (fx1 + label_size[0], fy2 + label_size[1] + 4),
                                        face_color, -1)
                            
                            # ラベルテキスト
                            cv2.putText(annotated, ag_label, (fx1, fy2 + label_size[1]),
                                      self.font, 0.4, (255, 255, 255), 1)
        
        # 基本的な検出結果の場合（後方互換性）
        elif isinstance(results, list):
            return self.draw_detections(frame, results, show_center, show_confidence,
                                       show_boxes, show_labels)
        
        return annotated


class IntegratedFileWorker(QThread):
    """統合型ファイル処理ワーカー（顔検出・年齢性別推定対応）"""
    
    # Signals
    progress_updated = Signal(int, int, float, float)  # current, total, fps, elapsed
    frame_processed = Signal(int, float, list)  # frame_num, timestamp, detections
    processing_completed = Signal(dict)  # output files
    error_occurred = Signal(str)
    log_message = Signal(str)
    detection_results = Signal(dict)  # 詳細な検出結果
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.detector = None
        self.pipeline = None
        self.visualizer = IntegratedVisualizerFile()
        self.exporter = CoordinateExporter()
        
        self.is_running = False
        self.params = {}
        
        # 顔検出・年齢性別推定設定
        self.enable_face_detection = False
        self.enable_age_gender = False
        self.face_confidence = 0.8
        self.use_stable_pipeline = True
        
    def set_parameters(self, params: dict):
        """Set processing parameters"""
        self.params = params
        
        # 顔検出パラメータの取得
        self.enable_face_detection = params.get('enable_face_detection', False)
        self.enable_age_gender = params.get('enable_age_gender', False)
        self.face_confidence = params.get('face_confidence', 0.8)
        
        logger.info(f"Processing parameters set: {params}")
        logger.info(f"Face detection: {self.enable_face_detection}, Age/Gender: {self.enable_age_gender}")
        
    def run(self):
        """Main processing loop"""
        try:
            self.is_running = True
            self.log_message.emit("Initializing detector...")
            
            # Initialize detector or pipeline
            if self.enable_face_detection or self.enable_age_gender:
                # 統合パイプライン使用
                config = {
                    'person_model': self.params['model'],
                    'person_confidence': self.params['confidence'],
                    'enable_face_detection': self.enable_face_detection,
                    'enable_age_gender': self.enable_age_gender,
                    'face_confidence': self.face_confidence,
                    'face_in_person_only': True,
                    'use_gpu': False
                }
                self.pipeline = StableDetectionPipeline(config)
                self.log_message.emit("Initialized integrated pipeline with face detection")
            else:
                # 基本的な人物検出のみ
                self.detector = PersonDetector(
                    model_name=self.params['model'],
                    confidence_threshold=self.params['confidence']
                )
                self.log_message.emit("Initialized person detector")
            
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
            if self.pipeline:
                self.pipeline.reset()
            
    def _prepare_output_files(self, width: int, height: int, fps: float) -> dict:
        """Prepare output files"""
        output_files = {}
        input_path = Path(self.params['input_file'])
        output_dir = Path(self.params['output_dir'])
        base_name = input_path.stem
        
        # Video writer with H.264 compression
        if self.params['export_video']:
            video_format = self.params['video_format']
            
            # Add suffix for face detection
            suffix = "_detected"
            if self.enable_face_detection:
                suffix += "_faces"
            if self.enable_age_gender:
                suffix += "_ag"
                
            video_path = output_dir / f"{base_name}{suffix}{video_format}"
            
            # Use H.264 codec through VideoCodecManager
            writer = VideoCodecManager.create_video_writer(
                str(video_path),
                fps,
                (width, height),
                use_h264=True
            )
            
            if writer is None:
                raise RuntimeError(f"Failed to create video writer for {video_path}")
            
            output_files['video_writer'] = writer
            output_files['video_path'] = str(video_path)
            
            # Log codec info
            codec_info = VideoCodecManager.ensure_h264_support()
            if codec_info:
                self.log_message.emit(f"Output video: {video_path} (H.264 compression)")
            else:
                self.log_message.emit(f"Output video: {video_path} (fallback codec)")
            
        # Data file
        if self.params['export_data']:
            data_format = self.params['data_format']
            suffix = "_detections"
            if self.enable_face_detection:
                suffix += "_faces"
            data_path = output_dir / f"{base_name}{suffix}{data_format}"
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
            
            # Detect objects
            if self.pipeline:
                # パイプライン使用（顔検出あり）
                results = self.pipeline.process_frame(frame)
                detections = results['persons']
                
                # 詳細結果を送信
                self.detection_results.emit(results)
            else:
                # 基本検出器使用（人物のみ）
                detections = self.detector.detect(frame)
                results = {
                    'persons': detections,
                    'faces': [],
                    'statistics': {
                        'num_persons': len(detections),
                        'num_faces': 0
                    }
                }
            
            # Draw annotations if needed
            if self.params['export_video'] or self.params['export_frames']:
                if self.pipeline:
                    # 拡張描画（顔・年齢性別含む）
                    annotated_frame = self.visualizer.draw_integrated_detections(
                        frame,
                        results,
                        show_center=False,
                        show_confidence=self.params['draw_confidence'],
                        show_boxes=self.params['draw_boxes'],
                        show_labels=self.params['draw_labels'],
                        show_faces=self.enable_face_detection,
                        show_age_gender=self.enable_age_gender
                    )
                else:
                    # 基本描画（人物のみ）
                    annotated_frame = self.visualizer.draw_detections(
                        frame,
                        detections,
                        show_center=False,
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
                    'detections': [],
                    'faces': []
                }
                
                # Store person detections
                for detection in detections:
                    bbox = detection['bbox']
                    if isinstance(bbox, tuple):
                        bbox = list(bbox)
                    elif hasattr(bbox, 'tolist'):
                        bbox = bbox.tolist()
                    
                    det_data = {
                        'bbox': bbox,
                        'confidence': float(detection['confidence']),
                        'class': detection.get('class_name', 'person')
                    }
                    
                    # Add face data if available
                    if 'faces' in detection:
                        det_data['faces'] = []
                        for face in detection['faces']:
                            face_bbox = face['bbox']
                            if isinstance(face_bbox, tuple):
                                face_bbox = list(face_bbox)
                            elif hasattr(face_bbox, 'tolist'):
                                face_bbox = face_bbox.tolist()
                                
                            face_data = {
                                'bbox': face_bbox,
                                'confidence': float(face.get('confidence', 0)),
                                'age': face.get('age'),
                                'gender': face.get('gender'),
                                'stable': face.get('stable', False)
                            }
                            det_data['faces'].append(face_data)
                    
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
                num_faces = results.get('statistics', {}).get('num_faces', 0)
                if num_faces > 0:
                    objects = f"{len(detections)} person(s), {num_faces} face(s)"
                else:
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
        """Save detection data as CSV with face information"""
        with open(file_path, 'w', newline='') as f:
            writer = csv.writer(f)
            headers = ['Frame', 'Timestamp', 'Detection_ID', 'X1', 'Y1', 'X2', 'Y2', 
                      'Confidence', 'Class', 'Num_Faces']
            
            if self.enable_face_detection:
                headers.extend(['Face_ID', 'Face_X1', 'Face_Y1', 'Face_X2', 'Face_Y2', 
                               'Face_Confidence'])
            if self.enable_age_gender:
                headers.extend(['Age', 'Gender'])
                
            writer.writerow(headers)
            
            for frame_data in data:
                frame = frame_data['frame']
                timestamp = frame_data['timestamp']
                
                for i, detection in enumerate(frame_data['detections']):
                    bbox = detection['bbox']
                    faces = detection.get('faces', [])
                    
                    if faces:
                        for j, face in enumerate(faces):
                            face_bbox = face['bbox']
                            row = [
                                frame, timestamp, i,
                                bbox[0], bbox[1], bbox[2], bbox[3],
                                detection['confidence'],
                                detection['class'],
                                len(faces),
                                j,
                                face_bbox[0], face_bbox[1], face_bbox[2], face_bbox[3],
                                face.get('confidence', 0)
                            ]
                            
                            if self.enable_age_gender:
                                row.extend([
                                    face.get('age', 'N/A'),
                                    face.get('gender', 'N/A')
                                ])
                                
                            writer.writerow(row)
                    else:
                        # No faces detected
                        row = [
                            frame, timestamp, i,
                            bbox[0], bbox[1], bbox[2], bbox[3],
                            detection['confidence'],
                            detection['class'],
                            0
                        ]
                        
                        if self.enable_face_detection:
                            row.extend(['', '', '', '', '', ''])
                        if self.enable_age_gender:
                            row.extend(['', ''])
                            
                        writer.writerow(row)
                    
    def _save_xml(self, file_path: str, data: List[Dict]):
        """Save detection data as XML with face information"""
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
                
                # Add face information
                faces = detection.get('faces', [])
                if faces:
                    faces_elem = ET.SubElement(det_elem, 'faces')
                    for face in faces:
                        face_elem = ET.SubElement(faces_elem, 'face')
                        face_elem.set('confidence', f"{face.get('confidence', 0):.3f}")
                        
                        if face.get('age'):
                            face_elem.set('age', str(face['age']))
                        if face.get('gender'):
                            face_elem.set('gender', face['gender'])
                        if face.get('stable'):
                            face_elem.set('stable', str(face['stable']))
                            
                        face_bbox_elem = ET.SubElement(face_elem, 'bbox')
                        face_bbox = face['bbox']
                        face_bbox_elem.set('x1', str(face_bbox[0]))
                        face_bbox_elem.set('y1', str(face_bbox[1]))
                        face_bbox_elem.set('x2', str(face_bbox[2]))
                        face_bbox_elem.set('y2', str(face_bbox[3]))
        
        tree = ET.ElementTree(root)
        tree.write(file_path, encoding='utf-8', xml_declaration=True)
        
    def stop(self):
        """Stop processing"""
        self.is_running = False
        self.wait(5000)  # Wait max 5 seconds