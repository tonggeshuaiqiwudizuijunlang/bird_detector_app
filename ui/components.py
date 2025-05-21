"""
UI组件模块 - 包含自定义Mac风格的按钮和框架
Creater Tz2H
"""

from PyQt5.QtWidgets import QPushButton, QFrame


class MacStyleButton(QPushButton):
    """Mac风格按钮"""

    def __init__(self, text, parent=None):
        """初始化Mac风格按钮"""
        super().__init__(text, parent)
        self.setStyleSheet("""
            QPushButton {
                background-color: rgba(0, 122, 255, 0.8);
                color: white;
                border: none;
                border-radius: 6px;
                padding: 8px 16px;
                font-size: 13px;
                font-weight: 500;
            }
            QPushButton:hover {
                background-color: rgba(0, 122, 255, 0.9);
            }
            QPushButton:pressed {
                background-color: rgba(0, 122, 255, 1.0);
            }
        """)


class MacStyleFrame(QFrame):
    """Mac风格框架"""

    def __init__(self, parent=None):
        """初始化Mac风格框架"""
        super().__init__(parent)
        self.setStyleSheet("""
            QFrame {
                background-color: rgba(255, 255, 255, 0.1);
                border-radius: 10px;
                border: 1px solid rgba(255, 255, 255, 0.2);
            }
        """)
