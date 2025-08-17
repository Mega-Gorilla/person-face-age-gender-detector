# Face Detection and Age/Gender Estimation Models Research (2025)

## Executive Summary

This document presents a comprehensive research on the latest state-of-the-art models for face detection and age/gender estimation as of 2025, based on recent benchmarks, GitHub repositories, and community recommendations.

## 1. Face Detection Models

### 1.1 Top Performers

#### **SCRFD (Sample and Computation Redistribution for Face Detection)**
- **Developer**: InsightFace team
- **Performance**: 
  - 3× faster than competitors on GPU
  - Up to 820 FPS on RTX4090
  - 3.86% better AP than TinaFace on WIDER Face Hard dataset
- **GitHub**: https://github.com/deepinsight/insightface
- **Strengths**: Excellent speed-accuracy trade-off, real-time performance
- **Weaknesses**: Issues with large faces (>40% of image) in original models

#### **RetinaFace**
- **Developer**: InsightFace team (CVPR 2020)
- **Performance**: 
  - Most universal detector
  - Excellent in crowds and varied lighting
  - Superior facial landmark detection
- **Strengths**: Robustness, accuracy, handles challenging scenarios
- **Weaknesses**: Slower than SCRFD (3× slower)

#### **YOLOv11-Face**
- **Performance**: Real-time detection with good accuracy
- **Strengths**: Fast inference, integrated with YOLO ecosystem
- **Use case**: When already using YOLO for person detection

### 1.2 Recommendation Matrix

| Priority | Model | Use Case |
|----------|-------|----------|
| Speed | SCRFD | Real-time applications, high FPS requirements |
| Accuracy | RetinaFace | Critical applications, challenging environments |
| Balance | SCRFD_10g_gnkps | General purpose with good speed/accuracy |

## 2. Age/Gender Estimation Models

### 2.1 State-of-the-Art Models

#### **MiVOLO (Multi-input Vision Transformer)**
- **Architecture**: Vision Transformer (ViT)
- **Performance**:
  - 971 FPS on NVIDIA V100 (batch=512)
  - Surpasses human-level accuracy
  - First place on UTKFace benchmark
  - State-of-the-art on IMDB dataset
- **GitHub**: https://github.com/WildChlamydia/MiVOLO
- **Key Features**:
  - Dual input (face + body)
  - Handles occluded faces
  - Unified model for both age and gender
- **Requirements**: PyTorch 1.13+, timm library

#### **SwinFace**
- **Architecture**: Swin Transformer
- **Performance**:
  - Age MAE: 0.22 on CLAP2015
  - Gender accuracy: 90.97% on RAF-DB
- **Features**: Multi-task (face recognition, expression, age, gender)

#### **DeepFace Library**
- **GitHub**: https://github.com/serengil/deepface
- **Features**: 
  - Wrapper for multiple detectors
  - Includes VGG-Face, FaceNet, ArcFace
  - Easy integration and comparison

### 2.2 Performance Benchmarks

| Model | Age MAE | Gender Accuracy | Speed |
|-------|---------|-----------------|-------|
| MiVOLO | 4.1 | 96% | 971 FPS |
| SwinFace | 0.22 (CLAP2015) | 90.97% | - |
| ResNet-50 | 6.55 (WIKI) | 95.02% (UTK) | - |
| VGG19 | - | 96.2% (Adience) | - |

## 3. Datasets for Evaluation

### Primary Datasets
1. **UTKFace**: 20,000+ images, ages 0-116
2. **IMDB-WIKI**: Large-scale with age/gender labels
3. **Adience**: 26K real-world unconstrained images
4. **LAGENDA**: 67,159 images with face+body pairs
5. **AFAD**: 160K+ Asian faces (largest for age estimation)

## 4. Implementation Recommendations

### 4.1 Optimal Pipeline for 2025

```
1. Person Detection: YOLOv11 (existing)
2. Face Detection: SCRFD_10g_gnkps (balance) or RetinaFace (accuracy)
3. Age/Gender: MiVOLO (best overall)
```

### 4.2 Key Considerations

- **Speed vs Accuracy**: SCRFD for speed, RetinaFace for accuracy
- **Robustness**: MiVOLO handles occlusions and partial faces
- **Integration**: DeepFace provides easy comparison framework
- **Real-time**: SCRFD + MiVOLO can achieve real-time performance

### 4.3 Technical Requirements

- **Hardware**: GPU recommended (NVIDIA with CUDA)
- **Framework**: PyTorch 1.13+ or TensorFlow 2.x
- **Memory**: ~2-4GB GPU memory for inference
- **Processing**: Batch processing for optimal throughput

## 5. Implementation Strategy

### Phase 1: Face Detection Integration
1. Integrate SCRFD for face detection
2. Fallback to RetinaFace for difficult cases
3. Benchmark against current setup

### Phase 2: Age/Gender Estimation
1. Implement MiVOLO for age/gender
2. Test on multiple datasets
3. Optimize for real-time performance

### Phase 3: Production Optimization
1. Model quantization for edge deployment
2. Batch processing optimization
3. Multi-threading for parallel processing

## 6. Expected Performance

Based on research, the recommended pipeline should achieve:
- **Face Detection**: >30 FPS on standard GPU
- **Age MAE**: <5 years
- **Gender Accuracy**: >95%
- **Overall Pipeline**: 15-20 FPS with all components

## 7. Ethical Considerations

- Gender prediction as binary classification has limitations
- Age estimation accuracy varies by ethnicity
- Privacy concerns with facial analysis
- Need for diverse training data

## 8. Conclusion

For the best balance of performance and accuracy in 2025:
- **Face Detection**: SCRFD (InsightFace)
- **Age/Gender**: MiVOLO (Vision Transformer)
- **Framework**: PyTorch-based implementation
- **Integration**: Modular pipeline with fallback options

These models represent the current state-of-the-art with proven performance on standard benchmarks and active community support.

## References

1. InsightFace GitHub: https://github.com/deepinsight/insightface
2. MiVOLO GitHub: https://github.com/WildChlamydia/MiVOLO
3. DeepFace GitHub: https://github.com/serengil/deepface
4. Papers with Code - Face Detection: https://paperswithcode.com/task/face-detection