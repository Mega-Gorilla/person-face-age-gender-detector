#!/usr/bin/env python3
"""
Test script for file processing mode with H.264 compression
"""

import sys
import os
import json
import time
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.detector import PersonDetector
from src.ui.visualizer import Visualizer
from src.utils.coordinate_exporter import CoordinateExporter
from src.core.common import format_detection_for_export, calculate_detection_stats
from src.utils.video_codec import VideoCodecManager
import cv2

def test_file_processing():
    """Test file processing with sample video"""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # Test video path
    video_path = project_root / "debug" / "sample_video" / "vtest(1).avi"
    
    if not video_path.exists():
        logger.error(f"Sample video not found: {video_path}")
        return False
    
    logger.info(f"Testing with video: {video_path}")
    
    # Output directory
    output_dir = project_root / "debug" / "output"
    output_dir.mkdir(exist_ok=True)
    
    try:
        # Initialize components
        logger.info("Initializing detector...")
        detector = PersonDetector(
            model_name="yolo11n.pt",
            confidence_threshold=0.5
        )
        
        visualizer = Visualizer()
        exporter = CoordinateExporter()
        
        # Open video
        logger.info("Opening video file...")
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        logger.info(f"Video info: {width}x{height} @ {fps:.1f}fps, {total_frames} frames")
        
        # Prepare output video with compression
        output_video_path = output_dir / "test_output.mp4"
        logger.info("Creating video writer with optimized codec...")
        out_video = VideoCodecManager.create_video_writer(
            str(output_video_path),
            fps,
            (width, height),
            use_h264=True  # Will use best available codec
        )
        
        if out_video is None:
            raise RuntimeError("Failed to create video writer")
        
        # Process frames
        logger.info("Processing frames...")
        frame_count = 0
        all_detections = []
        start_time = time.time()
        
        # Process every 5th frame for speed
        frame_skip = 5
        processed_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Skip frames
            if (frame_count - 1) % frame_skip != 0:
                continue
            
            processed_count += 1
            
            # Detect persons
            detections = detector.detect(frame)
            
            # Draw annotations
            annotated_frame = visualizer.draw_detections(
                frame,
                detections,
                show_center=False,
                show_confidence=True,
                show_boxes=True,
                show_labels=True
            )
            
            # Write frame
            out_video.write(annotated_frame)
            
            # Store detection data
            frame_data = {
                'frame': frame_count,
                'timestamp': frame_count / fps,
                'detections': [format_detection_for_export(d) for d in detections]
            }
            all_detections.append(frame_data)
            
            # Progress report
            if processed_count % 10 == 0:
                elapsed = time.time() - start_time
                progress = (frame_count / total_frames) * 100
                current_fps = processed_count / elapsed if elapsed > 0 else 0
                logger.info(f"Progress: {progress:.1f}% | FPS: {current_fps:.1f} | "
                          f"Frame {frame_count}/{total_frames}")
        
        # Cleanup
        cap.release()
        out_video.release()
        
        # Calculate statistics
        total_detections = sum(len(f['detections']) for f in all_detections)
        elapsed = time.time() - start_time
        
        logger.info(f"\n=== Processing Complete ===")
        logger.info(f"Processed frames: {processed_count}")
        logger.info(f"Total detections: {total_detections}")
        logger.info(f"Processing time: {elapsed:.2f}s")
        logger.info(f"Average FPS: {processed_count/elapsed:.1f}")
        
        # Export detection data
        logger.info("Exporting detection data...")
        
        # JSON export
        json_path = output_dir / "detections.json"
        exporter.export(all_detections, str(json_path), '.json')
        logger.info(f"JSON exported: {json_path}")
        
        # CSV export
        csv_path = output_dir / "detections.csv"
        exporter.export(all_detections, str(csv_path), '.csv')
        logger.info(f"CSV exported: {csv_path}")
        
        # Statistics
        stats = exporter.get_statistics(all_detections)
        logger.info(f"\n=== Detection Statistics ===")
        logger.info(f"Total frames: {stats['total_frames']}")
        logger.info(f"Frames with detections: {stats['frames_with_detections']}")
        logger.info(f"Total detections: {stats['total_detections']}")
        logger.info(f"Average detections per frame: {stats['avg_detections_per_frame']:.2f}")
        logger.info(f"Detection rate: {stats['detection_rate']:.2%}")
        
        if stats['total_detections'] > 0:
            logger.info(f"Average confidence: {stats['avg_confidence']:.2%}")
            logger.info(f"Min/Max confidence: {stats['min_confidence']:.2%} / {stats['max_confidence']:.2%}")
        
        # Verify outputs
        logger.info(f"\n=== Output Files ===")
        video_size_mb = output_video_path.stat().st_size / 1024 / 1024
        logger.info(f"Video: {output_video_path} ({video_size_mb:.1f} MB)")
        logger.info(f"JSON: {json_path} ({json_path.stat().st_size / 1024:.1f} KB)")
        logger.info(f"CSV: {csv_path} ({csv_path.stat().st_size / 1024:.1f} KB)")
        
        # Check compression
        original_size_mb = video_path.stat().st_size / 1024 / 1024
        logger.info(f"\n=== Compression Results ===")
        logger.info(f"Original video size: {original_size_mb:.2f} MB")
        logger.info(f"Output video size: {video_size_mb:.2f} MB")
        if video_size_mb < original_size_mb:
            reduction = ((original_size_mb - video_size_mb) / original_size_mb) * 100
            logger.info(f"✅ Size reduced by {reduction:.1f}%")
        else:
            increase = ((video_size_mb - original_size_mb) / original_size_mb) * 100
            logger.info(f"⚠️ Size increased by {increase:.1f}% (original was already compressed)")
        
        # Test sample frame
        if all_detections and all_detections[0]['detections']:
            sample = all_detections[0]['detections'][0]
            logger.info(f"\n=== Sample Detection ===")
            logger.info(f"BBox: {sample['bbox']}")
            logger.info(f"Confidence: {sample['confidence']:.2%}")
            logger.info(f"Class: {sample['class_name']}")
            logger.info(f"Center: {sample['center']}")
            logger.info(f"Area: {sample['area']}")
        
        logger.info("\n✅ File processing test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_common_functions():
    """Test common utility functions"""
    logger = logging.getLogger(__name__)
    logger.info("\n=== Testing Common Functions ===")
    
    # Test format_detection_for_export
    test_detection = {
        'bbox': (100, 200, 300, 400),
        'confidence': 0.95,
        'center': (200, 300),
        'area': 40000,
        'width': 200,
        'height': 200,
        'class_name': 'person',
        'class_id': 0
    }
    
    formatted = format_detection_for_export(test_detection)
    assert isinstance(formatted['bbox'], list), "BBox should be a list"
    assert formatted['bbox'] == [100, 200, 300, 400], "BBox values incorrect"
    assert formatted['confidence'] == 0.95, "Confidence incorrect"
    logger.info("✓ format_detection_for_export passed")
    
    # Test calculate_detection_stats
    detections = [
        {'confidence': 0.9, 'area': 1000},
        {'confidence': 0.8, 'area': 2000},
        {'confidence': 0.95, 'area': 1500}
    ]
    
    stats = calculate_detection_stats(detections)
    assert stats['count'] == 3, "Count incorrect"
    assert abs(stats['avg_confidence'] - 0.883) < 0.01, "Average confidence incorrect"
    assert stats['total_area'] == 4500, "Total area incorrect"
    logger.info("✓ calculate_detection_stats passed")
    
    logger.info("✅ All common function tests passed!")

if __name__ == "__main__":
    print("=" * 60)
    print("File Processing Test")
    print("=" * 60)
    
    # Test file processing
    success = test_file_processing()
    
    sys.exit(0 if success else 1)