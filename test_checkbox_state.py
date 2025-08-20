#!/usr/bin/env python3
"""
Test script to verify checkbox initial states
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.gui.workers.integrated_yolo_worker import IntegratedYoloWorker

def test_initial_states():
    """Test that checkboxes start with correct default values"""
    print("Testing initial checkbox states...")
    print("=" * 50)
    
    # Create worker instance
    worker = IntegratedYoloWorker()
    
    # Check face detection state
    print(f"Face detection enabled: {worker.enable_face_detection}")
    assert worker.enable_face_detection == False, "Face detection should be False by default"
    print("✓ Face detection is correctly disabled by default")
    
    # Check age/gender state
    print(f"Age/gender enabled: {worker.enable_age_gender}")
    assert worker.enable_age_gender == False, "Age/gender should be False by default"
    print("✓ Age/gender is correctly disabled by default")
    
    print("=" * 50)
    print("All tests passed! Default states are correct.")
    print("Users need to manually enable features as needed.")

if __name__ == "__main__":
    test_initial_states()