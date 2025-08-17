"""
Version information for YOLOv11 Person Detection System
"""

# Application metadata
APP_NAME = "YOLOv11 Person Detection System"
APP_SHORT_NAME = "YOLO11-PDS"

# Version information
MAJOR = 1
MINOR = 0
PATCH = 0
VERSION_STRING = f"{MAJOR}.{MINOR}.{PATCH}"

# Release information
RELEASE_DATE = "2025-08-17"
RELEASE_TYPE = "stable"  # stable, beta, alpha, dev

# Build information
BUILD_NUMBER = "20250817"
CODENAME = "Eagle Eye"

# Author information
AUTHOR = "Person Detection Team"
ORGANIZATION = "Detection Labs"
COPYRIGHT = f"© 2025 {ORGANIZATION}"

# Repository information
REPO_URL = "https://github.com/yourusername/person-face-age-gender-detector"
DOCS_URL = "https://github.com/yourusername/person-face-age-gender-detector/wiki"
ISSUES_URL = "https://github.com/yourusername/person-face-age-gender-detector/issues"

# Feature flags
FEATURES = {
    "stream_mode": True,
    "file_mode": True,
    "gpu_support": True,
    "multi_camera": False,
    "tracking": False,
    "face_detection": False,
    "age_gender": False
}

# Supported formats
SUPPORTED_VIDEO_FORMATS = [
    ".mp4", ".avi", ".mov", ".mkv", ".flv", 
    ".wmv", ".webm", ".m4v", ".mpg", ".mpeg"
]

SUPPORTED_IMAGE_FORMATS = [
    ".jpg", ".jpeg", ".png", ".bmp", 
    ".tiff", ".tif", ".webp"
]

EXPORT_FORMATS = {
    "video": [".mp4", ".avi"],
    "data": [".json", ".csv", ".xml"],
    "image": [".jpg", ".png"]
}

def get_full_version():
    """Get full version string with all details"""
    version = f"{APP_NAME} v{VERSION_STRING}"
    if RELEASE_TYPE != "stable":
        version += f" ({RELEASE_TYPE})"
    if CODENAME:
        version += f" - {CODENAME}"
    return version

def get_about_text():
    """Get formatted about text for dialogs"""
    return f"""
{APP_NAME}
Version: {VERSION_STRING} ({RELEASE_TYPE})
Build: {BUILD_NUMBER}
Release Date: {RELEASE_DATE}
Codename: {CODENAME}

{COPYRIGHT}

Powered by YOLOv11 (Ultralytics)
Built with PySide6 and OpenCV

Repository: {REPO_URL}
Documentation: {DOCS_URL}
Report Issues: {ISSUES_URL}
"""

def is_feature_enabled(feature_name):
    """Check if a specific feature is enabled"""
    return FEATURES.get(feature_name, False)

def get_supported_formats(format_type="video"):
    """Get list of supported formats by type"""
    if format_type == "video":
        return SUPPORTED_VIDEO_FORMATS
    elif format_type == "image":
        return SUPPORTED_IMAGE_FORMATS
    elif format_type == "export":
        return EXPORT_FORMATS
    else:
        return []