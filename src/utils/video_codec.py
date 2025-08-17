"""
Video codec utilities for H.264 encoding
"""

import cv2
import platform
import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)

class VideoCodecManager:
    """Manage video codecs for different platforms and formats"""
    
    # H.264 codec options for different platforms
    H264_CODECS = {
        'Windows': 'H264',     # DirectShow H.264
        'Darwin': 'avc1',      # macOS/iOS H.264 
        'Linux': 'h264',       # FFmpeg H.264
        'fallback': 'mp4v'     # MPEG-4 Part 2 as fallback
    }
    
    # Best compression codecs for different formats
    BEST_CODECS = {
        '.mp4': ['h264', 'H264', 'avc1', 'mp4v', 'XVID'],  # Try H.264 first, then fallbacks
        '.avi': ['XVID', 'DIVX', 'h264', 'mp4v'],  # XVID is best for AVI
        '.mov': ['h264', 'avc1', 'mp4v'],
        '.mkv': ['h264', 'XVID', 'mp4v'],
        '.webm': ['VP90', 'VP80'],  # VP9 provides better compression
    }
    
    # Alternative H.264 fourcc codes to try
    H264_ALTERNATIVES = [
        'H264',  # Standard H.264
        'h264',  # Lowercase variant
        'avc1',  # Apple's H.264
        'AVC1',  # Uppercase variant
        'x264',  # x264 encoder
        'X264',  # Uppercase x264
        'DAVC',  # Another H.264 variant
        'H.264', # With dot notation
    ]
    
    @classmethod
    def get_h264_codec(cls) -> int:
        """
        Get the best H.264 codec for the current platform
        
        Returns:
            fourcc code for H.264 codec
        """
        system = platform.system()
        
        # Try platform-specific codec first
        codec_string = cls.H264_CODECS.get(system, cls.H264_CODECS['fallback'])
        
        # Try to create fourcc code
        try:
            fourcc = cv2.VideoWriter_fourcc(*codec_string)
            logger.info(f"Using H.264 codec: {codec_string} for {system}")
            return fourcc
        except Exception as e:
            logger.warning(f"Failed to use {codec_string}: {e}")
            
        # Try alternatives
        for codec in cls.H264_ALTERNATIVES:
            try:
                if len(codec) == 4:
                    fourcc = cv2.VideoWriter_fourcc(*codec)
                else:
                    # Skip non-4-character codes
                    continue
                    
                # Test if codec is available
                if cls._test_codec(fourcc):
                    logger.info(f"Using alternative H.264 codec: {codec}")
                    return fourcc
            except Exception:
                continue
                
        # Fallback to MP4V
        logger.warning("H.264 codec not available, using MP4V fallback")
        return cv2.VideoWriter_fourcc(*'mp4v')
    
    @classmethod
    def _test_codec(cls, fourcc: int, test_size: Tuple[int, int] = (640, 480)) -> bool:
        """
        Test if a codec is available by trying to create a VideoWriter
        
        Args:
            fourcc: fourcc code to test
            test_size: test video dimensions
            
        Returns:
            True if codec is available
        """
        import tempfile
        import os
        
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
                tmp_path = tmp.name
                
            # Try to create VideoWriter
            writer = cv2.VideoWriter(tmp_path, fourcc, 30.0, test_size)
            
            if writer is None or not writer.isOpened():
                return False
                
            # Write a test frame
            test_frame = cv2.imread('/dev/null')
            if test_frame is None:
                # Create black frame for testing
                import numpy as np
                test_frame = np.zeros((test_size[1], test_size[0], 3), dtype=np.uint8)
                
            writer.write(test_frame)
            writer.release()
            
            # Check if file was created and has content
            success = os.path.exists(tmp_path) and os.path.getsize(tmp_path) > 0
            
            # Cleanup
            try:
                os.unlink(tmp_path)
            except:
                pass
                
            return success
            
        except Exception:
            return False
    
    @classmethod
    def get_codec_for_format(cls, file_extension: str) -> int:
        """
        Get appropriate codec for file format
        
        Args:
            file_extension: File extension (e.g., '.mp4', '.avi')
            
        Returns:
            fourcc code for codec
        """
        ext = file_extension.lower()
        
        # Get codec list for this format
        codec_list = cls.BEST_CODECS.get(ext, ['mp4v'])
        
        # Try each codec in order
        for codec_str in codec_list:
            if len(codec_str) != 4:
                continue
            
            try:
                fourcc = cv2.VideoWriter_fourcc(*codec_str)
                
                # Test if codec is available
                if cls._test_codec(fourcc):
                    logger.info(f"Using codec {codec_str} for format {ext}")
                    return fourcc
            except Exception:
                continue
        
        # Ultimate fallback to mp4v
        logger.warning(f"No optimal codec found for {ext}, using mp4v fallback")
        return cv2.VideoWriter_fourcc(*'mp4v')
    
    @classmethod
    def create_video_writer(cls, output_path: str, fps: float, 
                           frame_size: Tuple[int, int], 
                           use_h264: bool = True) -> Optional[cv2.VideoWriter]:
        """
        Create a VideoWriter with H.264 compression
        
        Args:
            output_path: Output video file path
            fps: Frames per second
            frame_size: (width, height) tuple
            use_h264: Whether to use H.264 codec
            
        Returns:
            VideoWriter object or None if failed
        """
        from pathlib import Path
        
        # Get file extension
        ext = Path(output_path).suffix
        
        # Get appropriate codec
        if use_h264:
            fourcc = cls.get_codec_for_format(ext)
        else:
            # Use default for format
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            
        # Create VideoWriter
        writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
        
        if writer.isOpened():
            logger.info(f"Created VideoWriter for {output_path} with codec {fourcc}")
            return writer
        else:
            logger.error(f"Failed to create VideoWriter for {output_path}")
            
            # Try fallback codec
            logger.info("Trying fallback codec MP4V")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
            
            if writer.isOpened():
                logger.info("Fallback codec MP4V successful")
                return writer
            else:
                logger.error("Fallback codec also failed")
                return None
    
    @classmethod
    def get_codec_info(cls, fourcc: int) -> str:
        """
        Get human-readable codec information
        
        Args:
            fourcc: fourcc code
            
        Returns:
            Codec name string
        """
        # Convert fourcc to string
        try:
            codec_chars = []
            for i in range(4):
                char = chr((fourcc >> (i * 8)) & 0xFF)
                codec_chars.append(char)
            return ''.join(codec_chars)
        except:
            return f"Unknown ({fourcc})"
    
    @classmethod
    def ensure_h264_support(cls) -> bool:
        """
        Check if H.264 support is available
        
        Returns:
            True if H.264 is supported
        """
        # Test H.264 variants
        h264_variants = ['h264', 'H264', 'avc1', 'AVC1', 'x264', 'X264']
        
        for codec_str in h264_variants:
            if len(codec_str) != 4:
                continue
            
            try:
                fourcc = cv2.VideoWriter_fourcc(*codec_str)
                if cls._test_codec(fourcc):
                    logger.info(f"H.264 codec is available ({codec_str})")
                    return True
            except Exception:
                continue
        
        # H.264 not available, but we have good fallbacks
        logger.info("H.264 codec not available, using optimized fallback codecs")
        logger.info("Current system supports: mp4v (MPEG-4), XVID, DIVX")
        logger.info("These codecs provide good compression for most use cases")
        
        return False