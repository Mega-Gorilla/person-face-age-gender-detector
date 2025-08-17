# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.1.0] - 2025-08-17 "Dual Vision"

### Added
- Tab interface with Stream and File modes
- File processing mode for video files
- Drag and drop support for video files
- Multiple export formats (JSON, CSV, XML) for detection data
- Centralized version management system
- Progress bar for file processing
- Batch processing capabilities

### Changed
- GUI now uses tab interface for better organization
- Updated to version 2.1.0
- Codename changed to "Dual Vision" (reflecting dual mode operation)
- English localization for all UI elements

### Fixed
- Visualizer parameter compatibility issues
- Window resize loop problem
- Text encoding issues in GUI

## [2.0.0] - 2025-08-17 "Eagle Eye"

### Added
- PySide6 GUI implementation
- Real-time statistics display
- Interactive controls (sliders, buttons)
- Screenshot functionality
- Model selection dropdown
- Camera settings configuration

### Changed
- Migrated from CUI to GUI/CUI dual mode
- Improved user experience with visual interface

## [1.0.0] - 2025-08-17

### Added
- Initial implementation with YOLOv11
- Real-time person detection from webcam
- OpenCV-based CUI interface
- Performance monitoring
- Modular architecture
- Multiple YOLOv11 model support (n, s, m, l, x)