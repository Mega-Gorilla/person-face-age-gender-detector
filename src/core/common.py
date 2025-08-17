"""
Common utilities for detection processing
"""

import numpy as np
from typing import Dict, List, Any, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

def format_detection_for_export(detection: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format detection data for export (JSON/CSV/XML)
    
    Args:
        detection: Raw detection dictionary from detector
        
    Returns:
        Formatted detection dictionary with consistent types
    """
    # Ensure bbox is a list
    bbox = detection.get('bbox', (0, 0, 0, 0))
    if isinstance(bbox, tuple):
        bbox = list(bbox)
    elif isinstance(bbox, np.ndarray):
        bbox = bbox.tolist()
    
    # Ensure center is a list
    center = detection.get('center', (0, 0))
    if isinstance(center, tuple):
        center = list(center)
    elif isinstance(center, np.ndarray):
        center = center.tolist()
    
    return {
        'bbox': bbox,
        'confidence': float(detection.get('confidence', 0.0)),
        'class_name': detection.get('class_name', 'person'),
        'class_id': int(detection.get('class_id', 0)),
        'center': center,
        'area': int(detection.get('area', 0)),
        'width': int(detection.get('width', 0)),
        'height': int(detection.get('height', 0))
    }

def calculate_detection_stats(detections: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate statistics from detection results
    
    Args:
        detections: List of detection dictionaries
        
    Returns:
        Statistics dictionary
    """
    if not detections:
        return {
            'count': 0,
            'avg_confidence': 0.0,
            'min_confidence': 0.0,
            'max_confidence': 0.0,
            'total_area': 0
        }
    
    confidences = [d.get('confidence', 0.0) for d in detections]
    areas = [d.get('area', 0) for d in detections]
    
    return {
        'count': len(detections),
        'avg_confidence': np.mean(confidences),
        'min_confidence': np.min(confidences),
        'max_confidence': np.max(confidences),
        'total_area': sum(areas),
        'avg_area': np.mean(areas) if areas else 0
    }

def validate_detection(detection: Dict[str, Any], 
                       min_area: int = 100,
                       min_confidence: float = 0.3) -> bool:
    """
    Validate if a detection meets minimum criteria
    
    Args:
        detection: Detection dictionary
        min_area: Minimum bounding box area
        min_confidence: Minimum confidence threshold
        
    Returns:
        True if detection is valid
    """
    area = detection.get('area', 0)
    confidence = detection.get('confidence', 0.0)
    
    return area >= min_area and confidence >= min_confidence

def merge_overlapping_detections(detections: List[Dict[str, Any]], 
                                 iou_threshold: float = 0.5) -> List[Dict[str, Any]]:
    """
    Merge overlapping detections based on IoU
    
    Args:
        detections: List of detection dictionaries
        iou_threshold: IoU threshold for merging
        
    Returns:
        Merged detections list
    """
    if len(detections) <= 1:
        return detections
    
    # Sort by confidence
    sorted_detections = sorted(detections, 
                              key=lambda x: x.get('confidence', 0.0), 
                              reverse=True)
    
    merged = []
    used = set()
    
    for i, det1 in enumerate(sorted_detections):
        if i in used:
            continue
            
        merged.append(det1)
        used.add(i)
        
        # Check for overlaps with remaining detections
        for j in range(i + 1, len(sorted_detections)):
            if j in used:
                continue
                
            det2 = sorted_detections[j]
            iou = calculate_iou(det1['bbox'], det2['bbox'])
            
            if iou > iou_threshold:
                used.add(j)
    
    return merged

def calculate_iou(bbox1: Tuple[int, int, int, int], 
                  bbox2: Tuple[int, int, int, int]) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes
    
    Args:
        bbox1: First bounding box (x1, y1, x2, y2)
        bbox2: Second bounding box (x1, y1, x2, y2)
        
    Returns:
        IoU value between 0 and 1
    """
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2
    
    # Calculate intersection
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i < x1_i or y2_i < y1_i:
        return 0.0
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    
    # Calculate union
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    
    if union == 0:
        return 0.0
    
    return intersection / union