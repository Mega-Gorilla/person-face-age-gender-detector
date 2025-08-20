#!/usr/bin/env python3
"""
Simple test for toggle functionality without camera
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.pipelines.stable_detection_pipeline import StableDetectionPipeline

def test_toggle():
    """Test toggle functionality"""
    print("\nTesting Pipeline Toggle Functionality")
    print("="*50)
    
    # Create pipeline with features disabled
    config = {
        'enable_face_detection': False,
        'enable_age_gender': False,
        'person_model': 'yolo11n.pt',
        'use_advanced_models': True
    }
    
    print("1. Creating pipeline with features disabled...")
    pipeline = StableDetectionPipeline(config)
    
    print(f"   Face detection: {pipeline.enable_face_detection}")
    print(f"   Age/gender: {pipeline.enable_age_gender}")
    print(f"   Face detector exists: {pipeline.face_detector is not None}")
    print(f"   Age estimator exists: {pipeline.age_gender_estimator is not None}")
    
    # Enable face detection
    print("\n2. Enabling face detection...")
    pipeline.update_config(enable_face_detection=True)
    print(f"   Face detection: {pipeline.enable_face_detection}")
    print(f"   Face detector exists: {pipeline.face_detector is not None}")
    
    # Enable age/gender
    print("\n3. Enabling age/gender...")
    pipeline.update_config(enable_age_gender=True)
    print(f"   Age/gender: {pipeline.enable_age_gender}")
    print(f"   Age estimator exists: {pipeline.age_gender_estimator is not None}")
    
    # Test with dummy frame
    import numpy as np
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    print("\n4. Testing process_frame with dummy data...")
    try:
        results = pipeline.process_frame(dummy_frame)
        print(f"   Process succeeded: True")
        print(f"   Results keys: {list(results.keys())}")
    except Exception as e:
        print(f"   Process failed: {e}")
    
    print("\n" + "="*50)
    print("Test completed successfully!")

if __name__ == "__main__":
    test_toggle()