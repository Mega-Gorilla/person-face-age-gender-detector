#!/usr/bin/env python3
"""
Test AVI compression with XVID codec
"""

import sys
import os
import cv2
import time
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

def test_avi_compression():
    """Test AVI compression with sample video"""
    
    sample_video = project_root / "debug" / "sample_video" / "vtest(1).avi"
    
    if not sample_video.exists():
        logger.error(f"Sample video not found: {sample_video}")
        return
    
    logger.info("=" * 60)
    logger.info("Testing AVI Compression with XVID Codec")
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
    
    # Test different output formats
    formats = [
        ('.avi', 'AVI with XVID'),
        ('.mp4', 'MP4 with best codec'),
    ]
    
    output_dir = project_root / "debug" / "output"
    output_dir.mkdir(exist_ok=True)
    
    results = {}
    
    for ext, description in formats:
        logger.info(f"\nProcessing {description}...")
        
        output_path = output_dir / f"sample_compressed{ext}"
        
        # Create writer with automatic codec selection
        writer = VideoCodecManager.create_video_writer(
            str(output_path),
            fps,
            (width, height),
            use_h264=True  # Will use best available codec
        )
        
        if writer is None:
            logger.error(f"Failed to create writer for {ext}")
            continue
        
        # Reset video to beginning
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        # Process frames
        start_time = time.time()
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            writer.write(frame)
            frame_count += 1
        
        writer.release()
        
        elapsed = time.time() - start_time
        output_size = output_path.stat().st_size / (1024 * 1024)
        
        results[ext] = {
            'size': output_size,
            'time': elapsed,
            'compression_ratio': input_size / output_size if output_size > 0 else 0,
            'size_reduction': ((input_size - output_size) / input_size * 100) if output_size > 0 else 0
        }
        
        logger.info(f"✅ {description} complete")
        logger.info(f"   Output size: {output_size:.2f} MB")
        logger.info(f"   Time: {elapsed:.2f}s")
        logger.info(f"   Compression ratio: {results[ext]['compression_ratio']:.2f}x")
        logger.info(f"   Size reduction: {results[ext]['size_reduction']:.1f}%")
    
    cap.release()
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Compression Summary")
    logger.info("=" * 60)
    logger.info(f"Original file: {input_size:.2f} MB")
    
    for ext, description in formats:
        if ext in results:
            r = results[ext]
            logger.info(f"\n{description}:")
            logger.info(f"  Size: {r['size']:.2f} MB")
            logger.info(f"  Compression: {r['compression_ratio']:.2f}x")
            if r['size_reduction'] > 0:
                logger.info(f"  ✅ Reduced size by {r['size_reduction']:.1f}%")
            else:
                logger.info(f"  ⚠️ File size increased by {-r['size_reduction']:.1f}%")
    
    # Recommendation
    best_ext = min(results.keys(), key=lambda x: results[x]['size']) if results else None
    if best_ext:
        logger.info(f"\n✅ Recommendation: Use {best_ext} format for best compression")
        logger.info(f"   Achieves {results[best_ext]['size_reduction']:.1f}% size reduction")

def main():
    """Main test function"""
    test_avi_compression()
    return 0

if __name__ == "__main__":
    sys.exit(main())