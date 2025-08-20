#!/usr/bin/env python3
"""
Script to download Caffe models for age/gender estimation
"""

import sys
import os
from pathlib import Path
import logging

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.core.age_gender_caffe import CaffeAgeGenderEstimator, check_gdown_installed, get_model_download_instructions

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Download models for age/gender estimation"""
    
    print("\n" + "="*60)
    print("CAFFE MODEL DOWNLOADER")
    print("="*60)
    print("\nThis script will download the Caffe models needed for")
    print("age and gender estimation (~90MB total).\n")
    
    # Check if gdown is installed
    print("1. Checking dependencies...")
    if not check_gdown_installed():
        print("   [ERROR] gdown is not installed")
        print("\n" + "="*60)
        print("INSTALLATION REQUIRED")
        print("="*60)
        print("\nThe 'gdown' package is required to download models from Google Drive.")
        print("\nPlease install it with:")
        print("   pip install gdown")
        print("\nThen run this script again.")
        print("="*60 + "\n")
        return 1
    
    print("   [OK] gdown is installed")
    
    # Check if models already exist
    model_dir = Path("models/age_gender_caffe")
    age_model = model_dir / "age_net.caffemodel"
    gender_model = model_dir / "gender_net.caffemodel"
    
    if age_model.exists() and gender_model.exists():
        print("\n2. Checking existing models...")
        print(f"   [OK] Age model found: {age_model}")
        print(f"   [OK] Gender model found: {gender_model}")
        
        response = input("\nModels already exist. Re-download? (y/N): ")
        if response.lower() != 'y':
            print("\nModels are already installed. Exiting.")
            return 0
    
    # Download models
    print("\n2. Downloading models...")
    print("   This may take a few minutes depending on your connection speed.\n")
    
    try:
        # Initialize estimator (this will trigger download)
        estimator = CaffeAgeGenderEstimator(use_gpu=False)
        
        if estimator.method == 'caffe':
            print("\n" + "="*60)
            print("SUCCESS!")
            print("="*60)
            print("\n[SUCCESS] Models downloaded and loaded successfully!")
            print(f"[SUCCESS] Models saved to: {model_dir}")
            print("\nYou can now use age and gender estimation in the application.")
            print("="*60 + "\n")
            return 0
        else:
            print("\n" + "="*60)
            print("DOWNLOAD INCOMPLETE")
            print("="*60)
            print("\n[WARNING] Models could not be downloaded or loaded properly.")
            print("\nPlease try:")
            print("1. Check your internet connection")
            print("2. Try running the script again")
            print("3. Or download manually:")
            print(get_model_download_instructions())
            print("="*60 + "\n")
            return 1
            
    except KeyboardInterrupt:
        print("\n\n[WARNING] Download interrupted by user.")
        print("Run this script again to resume/retry.")
        return 1
    except Exception as e:
        print(f"\n[ERROR] Error during download: {e}")
        print("\nTroubleshooting:")
        print("1. Check your internet connection")
        print("2. Make sure you have write permissions to the models directory")
        print("3. Try running with: python download_models.py")
        return 1


if __name__ == "__main__":
    sys.exit(main())