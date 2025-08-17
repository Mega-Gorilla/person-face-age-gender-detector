#!/usr/bin/env python3
"""
Test available video codecs on the system
"""

import sys
import os
import cv2
import numpy as np
import tempfile
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_codec(fourcc_str, name):
    """Test if a codec works"""
    try:
        # Create temp file
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
            tmp_path = tmp.name
        
        # Try to create writer
        fourcc = cv2.VideoWriter_fourcc(*fourcc_str) if len(fourcc_str) == 4 else -1
        writer = cv2.VideoWriter(tmp_path, fourcc, 30.0, (640, 480))
        
        if writer.isOpened():
            # Write test frame
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            writer.write(frame)
            writer.release()
            
            # Check file
            size = os.path.getsize(tmp_path)
            os.unlink(tmp_path)
            
            if size > 0:
                print(f"✅ {name:15} ({fourcc_str}): WORKING (size: {size} bytes)")
                return True
            else:
                print(f"❌ {name:15} ({fourcc_str}): File empty")
                return False
        else:
            print(f"❌ {name:15} ({fourcc_str}): Writer failed")
            try:
                os.unlink(tmp_path)
            except:
                pass
            return False
    except Exception as e:
        print(f"❌ {name:15} ({fourcc_str}): Error: {e}")
        return False

def main():
    print("=" * 60)
    print("Testing Available Video Codecs")
    print("=" * 60)
    
    # List of codecs to test
    codecs = [
        ('mp4v', 'MPEG-4 Part 2'),
        ('MP4V', 'MPEG-4 Part 2 (upper)'),
        ('XVID', 'Xvid'),
        ('MJPG', 'Motion JPEG'),
        ('X264', 'x264 H.264'),
        ('H264', 'H.264'),
        ('h264', 'H.264 (lower)'),
        ('avc1', 'H.264 AVC1'),
        ('AVC1', 'H.264 AVC1 (upper)'),
        ('mp4a', 'MPEG-4 Audio'),
        ('FMP4', 'FFmpeg MPEG-4'),
        ('DIV3', 'DivX 3'),
        ('DIVX', 'DivX'),
        ('U263', 'H.263'),
        ('I263', 'Intel H.263'),
        ('FLV1', 'Flash Video'),
        ('VP80', 'VP8'),
        ('VP90', 'VP9'),
    ]
    
    working_codecs = []
    
    for fourcc_str, name in codecs:
        if test_codec(fourcc_str, name):
            working_codecs.append((fourcc_str, name))
    
    print("\n" + "=" * 60)
    print(f"Summary: {len(working_codecs)}/{len(codecs)} codecs working")
    print("=" * 60)
    
    if working_codecs:
        print("\nWorking codecs:")
        for fourcc, name in working_codecs:
            print(f"  - {name} ({fourcc})")
    
    # Recommend best codec for MP4
    print("\n" + "=" * 60)
    print("Recommendation for MP4 files:")
    print("=" * 60)
    
    if any(c[0] in ['mp4v', 'MP4V'] for c in working_codecs):
        print("✅ Use MP4V codec (widely compatible)")
    elif any(c[0] == 'XVID' for c in working_codecs):
        print("✅ Use XVID codec (good compression)")
    elif any(c[0] == 'MJPG' for c in working_codecs):
        print("⚠️ Use MJPG codec (larger files)")
    else:
        print("❌ No suitable codec found for MP4")

if __name__ == "__main__":
    main()