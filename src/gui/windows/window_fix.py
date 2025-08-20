"""
Window management fixes for cross-platform compatibility
"""

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QMainWindow, QApplication
from PySide6.QtGui import QScreen, QKeySequence, QShortcut
import sys
import os
import platform
import logging

logger = logging.getLogger(__name__)

class WaylandWindowMixin:
    """Mixin class to fix window management issues on Wayland"""
    
    def setup_window_fixes(self):
        """Setup window management fixes for cross-platform compatibility"""
        
        # Detect platform and session type
        window_info = self._detect_window_environment()
        
        if window_info['platform'] == 'Windows':
            logger.debug("Windows platform detected")
            # Windows特有の設定は特に不要
        elif window_info['is_wayland']:
            logger.debug("Applying Wayland-specific window fixes")
            self._apply_wayland_fixes()
        elif window_info['is_x11']:
            logger.debug("X11 session detected")
        
        # Add universal window management shortcuts
        self._setup_window_shortcuts()
    
    def _detect_window_environment(self):
        """Detect window environment and platform"""
        info = {
            'platform': platform.system(),
            'session_type': os.environ.get('XDG_SESSION_TYPE', 'unknown'),
            'desktop_session': os.environ.get('DESKTOP_SESSION', 'unknown'),
            'xdg_current_desktop': os.environ.get('XDG_CURRENT_DESKTOP', 'unknown'),
            'gdm_session': os.environ.get('GDMSESSION', 'unknown'),
            'is_wayland': False,
            'is_x11': False,
            'is_gnome': False,
            'is_kde': False,
            'is_ubuntu': False
        }
        
        # Platform specific detection
        if info['platform'] == 'Linux':
            info['is_wayland'] = info['session_type'] == 'wayland'
            info['is_x11'] = info['session_type'] == 'x11'
            info['is_gnome'] = 'gnome' in info['desktop_session'].lower() or 'gnome' in info['xdg_current_desktop'].lower()
            info['is_kde'] = 'kde' in info['desktop_session'].lower() or 'kde' in info['xdg_current_desktop'].lower()
            info['is_ubuntu'] = 'ubuntu' in info['desktop_session'].lower()
        
        logger.debug(f"Window environment: {info}")
        return info
        
    def _apply_wayland_fixes(self):
        """Apply Wayland-specific fixes"""
        
        # Set window flags that work better on Wayland
        # Keep the default window flags but ensure maximize is allowed
        current_flags = self.windowFlags()
        
        # Ensure window can be maximized
        self.setWindowFlags(current_flags | Qt.WindowMaximizeButtonHint)
        
        # Set minimum size to prevent issues
        self.setMinimumSize(800, 600)
        
        # Use a timer to apply maximize state after window is shown
        if hasattr(self, '_should_maximize'):
            QTimer.singleShot(100, self._delayed_maximize)
    
    def _delayed_maximize(self):
        """Delayed maximize to work around Wayland timing issues"""
        if hasattr(self, '_should_maximize') and self._should_maximize:
            self.showMaximized()
            del self._should_maximize
    
    def _setup_window_shortcuts(self):
        """Setup keyboard shortcuts for window management"""
        
        # F11 for fullscreen toggle (already exists, ensure it works)
        fullscreen_shortcut = QShortcut(QKeySequence("F11"), self)
        fullscreen_shortcut.activated.connect(self.toggle_fullscreen_safe)
        
        # Ctrl+M for maximize toggle
        maximize_shortcut = QShortcut(QKeySequence("Ctrl+M"), self)
        maximize_shortcut.activated.connect(self.toggle_maximize)
        
        # Alt+Enter as alternative fullscreen toggle
        alt_fullscreen = QShortcut(QKeySequence("Alt+Return"), self)
        alt_fullscreen.activated.connect(self.toggle_fullscreen_safe)
        
        # Ctrl+Shift+M for window state cycle
        cycle_shortcut = QShortcut(QKeySequence("Ctrl+Shift+M"), self)
        cycle_shortcut.activated.connect(self.cycle_window_state)
        
        logger.debug("Window management shortcuts initialized:")
        logger.debug("  F11 / Alt+Enter: Toggle fullscreen")
        logger.debug("  Ctrl+M: Toggle maximize")
        logger.debug("  Ctrl+Shift+M: Cycle window states")
    
    def toggle_maximize(self):
        """Toggle between maximized and normal window state"""
        if self.isMaximized():
            self.showNormal()
            logger.debug("Window restored to normal")
        else:
            self.showMaximized()
            logger.debug("Window maximized")
    
    def toggle_fullscreen_safe(self):
        """Safe fullscreen toggle that works on Wayland"""
        if self.isFullScreen():
            self.showNormal()
            # On Wayland, we might need to re-apply maximize if it was maximized before
            if hasattr(self, '_was_maximized') and self._was_maximized:
                QTimer.singleShot(100, self.showMaximized)
            logger.debug("Exited fullscreen")
        else:
            # Store current maximized state
            self._was_maximized = self.isMaximized()
            self.showFullScreen()
            logger.debug("Entered fullscreen")
    
    def cycle_window_state(self):
        """Cycle through window states: Normal -> Maximized -> Fullscreen -> Normal"""
        if self.isFullScreen():
            self.showNormal()
            logger.debug("Window state: Normal")
        elif self.isMaximized():
            self.showFullScreen()
            logger.debug("Window state: Fullscreen")
        else:
            self.showMaximized()
            logger.debug("Window state: Maximized")
    
    def smart_maximize(self):
        """Smart maximize that uses available screen geometry"""
        screen = QApplication.primaryScreen()
        if screen:
            available_geometry = screen.availableGeometry()
            self.setGeometry(available_geometry)
            logger.debug(f"Window set to available geometry: {available_geometry}")
    
    def center_window(self):
        """Center the window on the screen"""
        screen = QApplication.primaryScreen()
        if screen:
            screen_geometry = screen.geometry()
            window_geometry = self.frameGeometry()
            
            # Calculate center position
            x = (screen_geometry.width() - window_geometry.width()) // 2
            y = (screen_geometry.height() - window_geometry.height()) // 2
            
            # Note: move() may not work on Wayland, but we try anyway
            self.move(x, y)
            logger.debug(f"Window centered at ({x}, {y})")
    
    def ensure_visible(self):
        """Ensure the window is visible on screen"""
        screen = QApplication.primaryScreen()
        if screen:
            screen_geometry = screen.geometry()
            window_geometry = self.geometry()
            
            # Check if window is outside screen bounds
            if not screen_geometry.contains(window_geometry):
                # Move window to be fully visible
                x = max(0, min(window_geometry.x(), 
                              screen_geometry.width() - window_geometry.width()))
                y = max(0, min(window_geometry.y(), 
                              screen_geometry.height() - window_geometry.height()))
                
                self.move(x, y)
                logger.debug("Window moved to ensure visibility")
    
    def showEvent(self, event):
        """Override showEvent to apply fixes when window is shown"""
        super().showEvent(event)
        
        # Ensure window is visible on first show
        if not hasattr(self, '_first_show_done'):
            self._first_show_done = True
            self.ensure_visible()
            
            # Apply any pending maximize state
            if hasattr(self, '_should_maximize'):
                QTimer.singleShot(100, self._delayed_maximize)


def get_window_manager_info():
    """Get information about the current window manager"""
    info = {
        'session_type': os.environ.get('XDG_SESSION_TYPE', 'unknown'),
        'desktop_session': os.environ.get('DESKTOP_SESSION', 'unknown'),
        'xdg_current_desktop': os.environ.get('XDG_CURRENT_DESKTOP', 'unknown'),
        'gdm_session': os.environ.get('GDMSESSION', 'unknown'),
    }
    
    # Check if running under Wayland
    info['is_wayland'] = info['session_type'] == 'wayland'
    
    # Check if running under X11
    info['is_x11'] = info['session_type'] == 'x11'
    
    # Detect specific desktop environments
    current_desktop = info['xdg_current_desktop'].lower()
    info['is_gnome'] = 'gnome' in current_desktop
    info['is_kde'] = 'kde' in current_desktop
    info['is_ubuntu'] = 'ubuntu' in current_desktop
    
    return info


def apply_platform_specific_fixes(window: QMainWindow):
    """Apply platform-specific fixes to a QMainWindow"""
    wm_info = get_window_manager_info()
    
    logger.debug(f"Window manager info: {wm_info}")
    
    if wm_info['is_wayland']:
        logger.debug("Applying Wayland-specific configuration")
        
        # On Wayland, certain window hints work better
        window.setWindowFlags(
            Qt.Window |
            Qt.WindowTitleHint |
            Qt.WindowSystemMenuHint |
            Qt.WindowMinimizeButtonHint |
            Qt.WindowMaximizeButtonHint |
            Qt.WindowCloseButtonHint
        )
        
        # Set a reasonable default size
        window.resize(1400, 900)
        
    elif wm_info['is_x11']:
        logger.debug("Running on X11 - standard window management should work")
    
    # Ubuntu-specific fixes
    if wm_info['is_ubuntu']:
        logger.debug("Applying Ubuntu-specific fixes")
        # Ubuntu with GNOME may need specific handling
        if wm_info['is_gnome']:
            # Ensure decorations are enabled
            window.setAttribute(Qt.WA_X11NetWmWindowTypeDesktop, False)