#!/usr/bin/env python3
"""
Test H.264 video compression functionality
"""

import sys
import os
import cv2
import time
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.video_codec import VideoCodecManager
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_h264_support():
    """Test if H.264 codec is supported"""
    logger.info("=" * 60)
    logger.info("Testing H.264 Codec Support")
    logger.info("=" * 60)
    
    # Check H.264 support
    is_supported = VideoCodecManager.ensure_h264_support()
    
    if is_supported:
        logger.info("✅ H.264 codec is supported on this system")
    else:
        logger.warning("⚠️ H.264 codec not available, using fallback codec")
    
    # Get codec for different formats
    formats = ['.mp4', '.avi', '.mov', '.mkv']
    for fmt in formats:
        fourcc = VideoCodecManager.get_codec_for_format(fmt)
        codec_name = VideoCodecManager.get_codec_info(fourcc)
        logger.info(f"Format {fmt}: {codec_name}")
    
    return is_supported

def create_test_video(output_path: str, use_h264: bool = True, 
                     num_frames: int = 100, fps: float = 30.0):
    """Create a test video with synthetic frames"""
    
    width, height = 640, 480
    
    logger.info(f"\nCreating test video: {output_path}")
    logger.info(f"Settings: {width}x{height}, {fps} fps, {num_frames} frames")
    logger.info(f"Using H.264: {use_h264}")
    
    # Create video writer
    writer = VideoCodecManager.create_video_writer(
        output_path,
        fps,
        (width, height),
        use_h264=use_h264
    )
    
    if writer is None:
        logger.error("Failed to create video writer")
        return False
    
    # Generate and write frames
    start_time = time.time()
    
    for i in range(num_frames):
        # Create a frame with moving rectangle
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Add some visual content
        x = int((width - 100) * (i / num_frames))
        y = int(height / 2 - 50)
        cv2.rectangle(frame, (x, y), (x + 100, y + 100), (0, 255, 0), -1)
        
        # Add frame number
        cv2.putText(frame, f"Frame {i+1}/{num_frames}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   1, (255, 255, 255), 2)
        
        # Add codec info
        if use_h264:
            cv2.putText(frame, "H.264 Compression", 
                       (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (0, 255, 255), 2)
        else:
            cv2.putText(frame, "Standard Compression", 
                       (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (255, 255, 0), 2)
        
        writer.write(frame)
    
    writer.release()
    
    elapsed = time.time() - start_time
    file_size = Path(output_path).stat().st_size / (1024 * 1024)  # MB
    
    logger.info(f"✅ Video created successfully")
    logger.info(f"   Time: {elapsed:.2f}s")
    logger.info(f"   File size: {file_size:.2f} MB")
    logger.info(f"   Size per frame: {file_size * 1024 / num_frames:.2f} KB")
    
    return True

def compare_compression():
    """Compare H.264 vs standard compression"""
    logger.info("\n" + "=" * 60)
    logger.info("Comparing H.264 vs Standard Compression")
    logger.info("=" * 60)
    
    output_dir = project_root / "debug" / "output"
    output_dir.mkdir(exist_ok=True)
    
    # Test videos with same content
    num_frames = 150
    fps = 30.0
    
    # Create H.264 compressed video
    h264_path = str(output_dir / "test_h264.mp4")
    if create_test_video(h264_path, use_h264=True, 
                        num_frames=num_frames, fps=fps):
        h264_size = Path(h264_path).stat().st_size / (1024 * 1024)
    else:
        h264_size = 0
    
    # Create standard compressed video
    standard_path = str(output_dir / "test_standard.mp4")
    if create_test_video(standard_path, use_h264=False, 
                        num_frames=num_frames, fps=fps):
        standard_size = Path(standard_path).stat().st_size / (1024 * 1024)
    else:
        standard_size = 0
    
    # Compare results
    if h264_size > 0 and standard_size > 0:
        logger.info("\n" + "=" * 60)
        logger.info("Compression Comparison Results")
        logger.info("=" * 60)
        logger.info(f"H.264 video size:    {h264_size:.2f} MB")
        logger.info(f"Standard video size: {standard_size:.2f} MB")
        
        if h264_size < standard_size:
            reduction = ((standard_size - h264_size) / standard_size) * 100
            logger.info(f"✅ H.264 is {reduction:.1f}% smaller!")
        else:
            logger.info("Standard compression performed better")
        
        logger.info(f"\nOutput files saved to: {output_dir}")

def test_with_sample_video():
    """Test H.264 compression with actual sample video"""
    sample_video = project_root / "debug" / "sample_video" / "vtest(1).avi"
    
    if not sample_video.exists():
        logger.warning(f"Sample video not found: {sample_video}")
        return
    
    logger.info("\n" + "=" * 60)
    logger.info("Testing with Sample Video")
    logger.info("=" * 60)
    
    # Open input video
    cap = cv2.VideoCapture(str(sample_video))
    if not cap.isOpened():
        logger.error("Failed to open sample video")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    logger.info(f"Input video: {width}x{height} @ {fps:.1f} fps, {total_frames} frames")
    input_size = sample_video.stat().st_size / (1024 * 1024)
    logger.info(f"Input size: {input_size:.2f} MB")
    
    # Create output with H.264
    output_dir = project_root / "debug" / "output"
    output_path = output_dir / "sample_h264.mp4"
    
    writer = VideoCodecManager.create_video_writer(
        str(output_path),
        fps,
        (width, height),
        use_h264=True
    )
    
    if writer is None:
        logger.error("Failed to create video writer")
        cap.release()
        return
    
    # Process frames
    logger.info("Processing frames...")
    frame_count = 0
    start_time = time.time()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        writer.write(frame)
        frame_count += 1
        
        if frame_count % 100 == 0:
            logger.info(f"  Processed {frame_count}/{total_frames} frames")
    
    cap.release()
    writer.release()
    
    elapsed = time.time() - start_time
    output_size = output_path.stat().st_size / (1024 * 1024)
    
    logger.info(f"\n✅ Processing complete")
    logger.info(f"   Time: {elapsed:.2f}s")
    logger.info(f"   Output size: {output_size:.2f} MB")
    logger.info(f"   Compression ratio: {input_size/output_size:.2f}x")
    logger.info(f"   Size reduction: {((input_size - output_size)/input_size * 100):.1f}%")
    logger.info(f"   Output saved to: {output_path}")

def main():
    """Main test function"""
    print("=" * 60)
    print("H.264 Video Compression Test Suite")
    print("=" * 60)
    
    # Test H.264 support
    h264_supported = test_h264_support()
    
    # Compare compression methods
    compare_compression()
    
    # Test with sample video if available
    test_with_sample_video()
    
    print("\n✅ All tests completed!")
    return 0

if __name__ == "__main__":
    sys.exit(main())