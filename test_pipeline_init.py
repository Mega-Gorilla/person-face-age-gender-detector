#!/usr/bin/env python3
"""
Test pipeline initialization and toggle functionality
"""

import sys
import logging
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

from src.gui.workers.integrated_yolo_worker import IntegratedYoloWorker
from src.pipelines.stable_detection_pipeline import StableDetectionPipeline

def test_pipeline_initialization():
    """Test that pipeline initializes correctly even with features disabled"""
    print("\n" + "="*60)
    print("Testing Pipeline Initialization")
    print("="*60)
    
    # Test 1: Worker initialization
    print("\n1. Creating worker with defaults (face=False, age_gender=False)...")
    worker = IntegratedYoloWorker()
    print(f"   Face detection: {worker.enable_face_detection}")
    print(f"   Age/gender: {worker.enable_age_gender}")
    
    # Test 2: Initialize components
    print("\n2. Initializing components...")
    success = worker.initialize_components()
    print(f"   Initialization success: {success}")
    print(f"   Pipeline exists: {worker.pipeline is not None}")
    
    if worker.pipeline:
        print(f"   Pipeline face detection: {worker.pipeline.enable_face_detection}")
        print(f"   Pipeline age/gender: {worker.pipeline.enable_age_gender}")
        print(f"   Face detector exists: {worker.pipeline.face_detector is not None}")
        print(f"   Age/gender estimator exists: {worker.pipeline.age_gender_estimator is not None}")
    
    # Test 3: Toggle face detection
    print("\n3. Toggling face detection ON...")
    worker.toggle_face_detection(True)
    print(f"   Worker face detection: {worker.enable_face_detection}")
    if worker.pipeline:
        print(f"   Pipeline face detection: {worker.pipeline.enable_face_detection}")
    
    # Test 4: Toggle age/gender
    print("\n4. Toggling age/gender ON...")
    worker.toggle_age_gender(True)
    print(f"   Worker age/gender: {worker.enable_age_gender}")
    if worker.pipeline:
        print(f"   Pipeline age/gender: {worker.pipeline.enable_age_gender}")
    
    print("\n" + "="*60)
    print("RESULT: Pipeline initialization and toggle tests completed")
    print("="*60)
    
    # Cleanup
    if worker.camera:
        worker.camera.release()
    
    return success

def test_direct_pipeline():
    """Test pipeline directly"""
    print("\n" + "="*60)
    print("Testing Direct Pipeline Creation")
    print("="*60)
    
    config = {
        'enable_face_detection': False,
        'enable_age_gender': False,
        'use_advanced_models': True
    }
    
    print("\n1. Creating pipeline with features disabled...")
    pipeline = StableDetectionPipeline(config)
    
    print(f"   Face detection enabled: {pipeline.enable_face_detection}")
    print(f"   Age/gender enabled: {pipeline.enable_age_gender}")
    print(f"   Face detector exists: {pipeline.face_detector is not None}")
    print(f"   Age/gender estimator exists: {pipeline.age_gender_estimator is not None}")
    
    # Toggle features
    print("\n2. Enabling features via update_config...")
    pipeline.update_config(enable_face_detection=True, enable_age_gender=True)
    
    print(f"   Face detection enabled: {pipeline.enable_face_detection}")
    print(f"   Age/gender enabled: {pipeline.enable_age_gender}")
    
    print("\n" + "="*60)
    print("RESULT: Direct pipeline test completed")
    print("="*60)

if __name__ == "__main__":
    print("Testing pipeline initialization with disabled features...")
    
    # Test direct pipeline
    test_direct_pipeline()
    
    # Test worker with pipeline
    test_pipeline_initialization()
    
    print("\nAll tests completed!")