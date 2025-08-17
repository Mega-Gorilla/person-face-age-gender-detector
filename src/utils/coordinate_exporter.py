"""
Coordinate data exporter for detection results
"""

import json
import csv
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
import logging

logger = logging.getLogger(__name__)

class CoordinateExporter:
    """Export detection coordinates in various formats"""
    
    def __init__(self):
        self.supported_formats = ['.json', '.csv', '.xml', '.txt']
        
    def export(self, data: List[Dict], output_path: str, format: Optional[str] = None):
        """Export detection data to file"""
        path = Path(output_path)
        
        # Determine format from extension if not specified
        if format is None:
            format = path.suffix.lower()
            
        if format not in self.supported_formats:
            raise ValueError(f"Unsupported format: {format}. Supported: {self.supported_formats}")
            
        # Create output directory if needed
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Export based on format
        if format == '.json':
            self.export_json(data, path)
        elif format == '.csv':
            self.export_csv(data, path)
        elif format == '.xml':
            self.export_xml(data, path)
        elif format == '.txt':
            self.export_txt(data, path)
            
        logger.info(f"Exported detection data to {path}")
        
    def export_json(self, data: List[Dict], output_path: Path):
        """Export as JSON format"""
        # Ensure data is JSON serializable
        serializable_data = self._make_serializable(data)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_data, f, indent=2, ensure_ascii=False)
            
    def export_csv(self, data: List[Dict], output_path: Path):
        """Export as CSV format"""
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Write header
            writer.writerow([
                'Frame', 'Timestamp', 'Detection_ID', 
                'X1', 'Y1', 'X2', 'Y2',
                'Width', 'Height', 'Center_X', 'Center_Y',
                'Confidence', 'Class', 'Area'
            ])
            
            # Write data
            for frame_data in data:
                frame = frame_data.get('frame', 0)
                timestamp = frame_data.get('timestamp', 0.0)
                
                for i, detection in enumerate(frame_data.get('detections', [])):
                    bbox = detection.get('bbox', [0, 0, 0, 0])
                    x1, y1, x2, y2 = bbox
                    width = x2 - x1
                    height = y2 - y1
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    area = width * height
                    
                    writer.writerow([
                        frame, f"{timestamp:.3f}", i,
                        x1, y1, x2, y2,
                        width, height, center_x, center_y,
                        f"{detection.get('confidence', 0.0):.3f}",
                        detection.get('class_name', detection.get('class', 'person')),
                        area
                    ])
                    
    def export_xml(self, data: List[Dict], output_path: Path):
        """Export as XML format"""
        root = ET.Element('detection_results')
        root.set('version', '1.0')
        
        # Add metadata
        metadata = ET.SubElement(root, 'metadata')
        ET.SubElement(metadata, 'total_frames').text = str(len(data))
        
        total_detections = sum(len(f.get('detections', [])) for f in data)
        ET.SubElement(metadata, 'total_detections').text = str(total_detections)
        
        # Add frames
        frames_elem = ET.SubElement(root, 'frames')
        
        for frame_data in data:
            frame_elem = ET.SubElement(frames_elem, 'frame')
            frame_elem.set('number', str(frame_data.get('frame', 0)))
            frame_elem.set('timestamp', f"{frame_data.get('timestamp', 0.0):.3f}")
            
            detections = frame_data.get('detections', [])
            frame_elem.set('detection_count', str(len(detections)))
            
            for detection in detections:
                det_elem = ET.SubElement(frame_elem, 'detection')
                det_elem.set('class', detection.get('class', 'unknown'))
                det_elem.set('confidence', f"{detection.get('confidence', 0.0):.3f}")
                
                # Bounding box
                bbox_elem = ET.SubElement(det_elem, 'bounding_box')
                bbox = detection.get('bbox', [0, 0, 0, 0])
                bbox_elem.set('x1', str(bbox[0]))
                bbox_elem.set('y1', str(bbox[1]))
                bbox_elem.set('x2', str(bbox[2]))
                bbox_elem.set('y2', str(bbox[3]))
                
                # Additional properties
                props_elem = ET.SubElement(det_elem, 'properties')
                width = bbox[2] - bbox[0]
                height = bbox[3] - bbox[1]
                ET.SubElement(props_elem, 'width').text = str(width)
                ET.SubElement(props_elem, 'height').text = str(height)
                ET.SubElement(props_elem, 'area').text = str(width * height)
                ET.SubElement(props_elem, 'center_x').text = str((bbox[0] + bbox[2]) / 2)
                ET.SubElement(props_elem, 'center_y').text = str((bbox[1] + bbox[3]) / 2)
                
        # Write to file
        tree = ET.ElementTree(root)
        tree.write(output_path, encoding='utf-8', xml_declaration=True)
        
    def export_txt(self, data: List[Dict], output_path: Path):
        """Export as plain text format"""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("YOLOv11 Person Detection Results\n")
            f.write("=" * 50 + "\n\n")
            
            total_frames = len(data)
            total_detections = sum(len(f.get('detections', [])) for f in data)
            
            f.write(f"Total Frames: {total_frames}\n")
            f.write(f"Total Detections: {total_detections}\n")
            f.write(f"Average Detections per Frame: {total_detections/total_frames:.2f}\n\n")
            
            f.write("-" * 50 + "\n")
            f.write("Frame-by-Frame Results:\n")
            f.write("-" * 50 + "\n\n")
            
            for frame_data in data:
                frame = frame_data.get('frame', 0)
                timestamp = frame_data.get('timestamp', 0.0)
                detections = frame_data.get('detections', [])
                
                f.write(f"Frame {frame} (Time: {timestamp:.3f}s)\n")
                
                if not detections:
                    f.write("  No detections\n")
                else:
                    f.write(f"  Detections: {len(detections)}\n")
                    for i, detection in enumerate(detections):
                        bbox = detection.get('bbox', [0, 0, 0, 0])
                        conf = detection.get('confidence', 0.0)
                        cls = detection.get('class', 'unknown')
                        
                        f.write(f"    [{i+1}] Class: {cls}, ")
                        f.write(f"Confidence: {conf:.3f}, ")
                        f.write(f"BBox: ({bbox[0]:.0f}, {bbox[1]:.0f}, {bbox[2]:.0f}, {bbox[3]:.0f})\n")
                        
                f.write("\n")
                
    def export_yolo_format(self, data: List[Dict], output_dir: Path, 
                           image_width: int, image_height: int):
        """Export in YOLO training format (one file per frame)"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for frame_data in data:
            frame = frame_data.get('frame', 0)
            detections = frame_data.get('detections', [])
            
            # Create label file for this frame
            label_file = output_dir / f"frame_{frame:06d}.txt"
            
            with open(label_file, 'w') as f:
                for detection in detections:
                    # YOLO format: class_id x_center y_center width height (normalized)
                    bbox = detection.get('bbox', [0, 0, 0, 0])
                    x1, y1, x2, y2 = bbox
                    
                    # Calculate normalized values
                    x_center = ((x1 + x2) / 2) / image_width
                    y_center = ((y1 + y2) / 2) / image_height
                    width = (x2 - x1) / image_width
                    height = (y2 - y1) / image_height
                    
                    # Class ID (0 for person in COCO dataset)
                    class_id = 0
                    
                    f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                    
    def export_coco_format(self, data: List[Dict], output_path: Path):
        """Export in COCO dataset format"""
        coco_data = {
            "info": {
                "description": "YOLOv11 Person Detection Results",
                "version": "1.0",
                "year": 2025
            },
            "images": [],
            "annotations": [],
            "categories": [
                {"id": 1, "name": "person", "supercategory": "person"}
            ]
        }
        
        annotation_id = 0
        
        for frame_data in data:
            frame = frame_data.get('frame', 0)
            
            # Add image info
            image_info = {
                "id": frame,
                "file_name": f"frame_{frame:06d}.jpg",
                "width": 0,  # Should be set from actual image
                "height": 0  # Should be set from actual image
            }
            coco_data["images"].append(image_info)
            
            # Add annotations
            for detection in frame_data.get('detections', []):
                bbox = detection.get('bbox', [0, 0, 0, 0])
                x1, y1, x2, y2 = bbox
                width = x2 - x1
                height = y2 - y1
                
                annotation = {
                    "id": annotation_id,
                    "image_id": frame,
                    "category_id": 1,  # person
                    "bbox": [x1, y1, width, height],  # COCO format: [x, y, width, height]
                    "area": width * height,
                    "segmentation": [],  # Empty for bounding box only
                    "iscrowd": 0,
                    "score": detection.get('confidence', 0.0)
                }
                coco_data["annotations"].append(annotation)
                annotation_id += 1
                
        # Save COCO format JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(coco_data, f, indent=2)
            
    def _make_serializable(self, data: Any) -> Any:
        """Convert numpy arrays and other non-serializable types to serializable format"""
        if isinstance(data, np.ndarray):
            return data.tolist()
        elif isinstance(data, (np.integer, np.floating)):
            return data.item()
        elif isinstance(data, (np.int64, np.int32, np.int16, np.int8)):
            return int(data)
        elif isinstance(data, (np.float64, np.float32, np.float16)):
            return float(data)
        elif isinstance(data, np.bool_):
            return bool(data)
        elif isinstance(data, dict):
            return {k: self._make_serializable(v) for k, v in data.items()}
        elif isinstance(data, (list, tuple)):
            return [self._make_serializable(item) for item in data]
        else:
            # Try to convert any remaining numpy types
            if hasattr(data, 'item'):
                return data.item()
            return data
            
    def merge_results(self, results_list: List[List[Dict]]) -> List[Dict]:
        """Merge multiple detection results"""
        merged = []
        for results in results_list:
            merged.extend(results)
            
        # Sort by frame number
        merged.sort(key=lambda x: x.get('frame', 0))
        return merged
        
    def filter_by_confidence(self, data: List[Dict], min_confidence: float) -> List[Dict]:
        """Filter detections by minimum confidence threshold"""
        filtered_data = []
        
        for frame_data in data:
            filtered_frame = frame_data.copy()
            filtered_detections = [
                d for d in frame_data.get('detections', [])
                if d.get('confidence', 0.0) >= min_confidence
            ]
            filtered_frame['detections'] = filtered_detections
            filtered_data.append(filtered_frame)
            
        return filtered_data
        
    def get_statistics(self, data: List[Dict]) -> Dict:
        """Calculate statistics from detection data"""
        total_frames = len(data)
        frames_with_detections = sum(1 for f in data if f.get('detections', []))
        total_detections = sum(len(f.get('detections', [])) for f in data)
        
        all_confidences = []
        for frame_data in data:
            for detection in frame_data.get('detections', []):
                all_confidences.append(detection.get('confidence', 0.0))
                
        stats = {
            'total_frames': total_frames,
            'frames_with_detections': frames_with_detections,
            'frames_without_detections': total_frames - frames_with_detections,
            'total_detections': total_detections,
            'avg_detections_per_frame': total_detections / total_frames if total_frames > 0 else 0,
            'detection_rate': frames_with_detections / total_frames if total_frames > 0 else 0,
        }
        
        if all_confidences:
            stats['avg_confidence'] = np.mean(all_confidences)
            stats['min_confidence'] = np.min(all_confidences)
            stats['max_confidence'] = np.max(all_confidences)
            stats['std_confidence'] = np.std(all_confidences)
        else:
            stats['avg_confidence'] = 0.0
            stats['min_confidence'] = 0.0
            stats['max_confidence'] = 0.0
            stats['std_confidence'] = 0.0
            
        return stats