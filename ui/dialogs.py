"""
对话框模块 - 包含设置对话框和密度图设置对话框
Creater Tz2H
"""

from PyQt5.QtWidgets import (
    QCheckBox,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)
from ultralytics import YOLO


class SettingsDialog(QDialog):
    """设置对话框"""

    def __init__(
        self, parent=None, model_path=None, all_classes=None, selected_classes=None
    ):
        """初始化设置对话框"""
        super().__init__(parent)
        self.setWindowTitle("设置")
        self.resize(400, 600)
        self.model_path = model_path
        self.all_classes = all_classes or []
        self.selected_classes = set(selected_classes) if selected_classes else set()
        self.result_model_path = model_path
        self.result_selected_classes = set(self.selected_classes)

        layout = QVBoxLayout(self)

        # 模型选择
        model_layout = QHBoxLayout()
        self.model_label = QLabel(self.model_path if self.model_path else "未选择模型")
        self.model_btn = QPushButton("选择模型")
        self.model_btn.clicked.connect(self.choose_model)
        model_layout.addWidget(self.model_label)
        model_layout.addWidget(self.model_btn)
        layout.addLayout(model_layout)

        # 类别复选框区域
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.class_widget = QWidget()
        self.class_layout = QVBoxLayout(self.class_widget)
        self.recog_checkboxes = []
        self.refresh_class_checkboxes()
        self.scroll.setWidget(self.class_widget)
        layout.addWidget(QLabel("请选择需要识别的类别："))
        layout.addWidget(self.scroll)

        # 确认按钮
        btn_layout = QHBoxLayout()
        ok_btn = QPushButton("确认")
        ok_btn.clicked.connect(self.accept)
        cancel_btn = QPushButton("取消")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(ok_btn)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)

    def choose_model(self):
        """选择模型文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择YOLO模型", "", "模型文件 (*.pt)"
        )
        if file_path:
            model = YOLO(file_path)
            self.model_path = file_path
            self.all_classes = list(model.names.values())
            self.result_model_path = file_path
            self.result_selected_classes = set(self.all_classes)  # 默认全选
            self.refresh_class_checkboxes()
            self.model_label.setText(file_path)

    def refresh_class_checkboxes(self):
        """刷新类别复选框"""
        # 清空原有
        for cb in self.recog_checkboxes:
            self.class_layout.removeWidget(cb)
            cb.deleteLater()
        self.recog_checkboxes = []
        # 重新生成
        for cls in self.all_classes:
            cb = QCheckBox(cls)
            cb.setChecked(cls in self.result_selected_classes)
            cb.stateChanged.connect(self.update_selected_classes)
            self.class_layout.addWidget(cb)
            self.recog_checkboxes.append(cb)

    def update_selected_classes(self):
        """更新选中的类别"""
        self.result_selected_classes = set()
        for cb in self.recog_checkboxes:
            if cb.isChecked():
                self.result_selected_classes.add(cb.text())

    def get_result(self):
        """获取对话框结果"""
        return self.result_model_path, self.result_selected_classes


class DensityDialog(QDialog):
    """密度图设置对话框"""

    def __init__(self, parent=None, density_classes=None):
        """初始化密度图设置对话框"""
        super().__init__(parent)
        self.setWindowTitle("密度图设置")
        self.resize(400, 600)
        self.density_classes = density_classes if density_classes else set()

        layout = QVBoxLayout(self)

        # 类别复选框区域
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.class_widget = QWidget()
        self.class_layout = QVBoxLayout(self.class_widget)
        self.density_checkboxes = []
        self.refresh_class_checkboxes()
        self.scroll.setWidget(self.class_widget)
        layout.addWidget(QLabel("请选择需要显示的类别："))
        layout.addWidget(self.scroll)

        # 确认按钮
        btn_layout = QHBoxLayout()
        ok_btn = QPushButton("确认")
        ok_btn.clicked.connect(self.accept)
        cancel_btn = QPushButton("取消")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(ok_btn)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)

    def refresh_class_checkboxes(self):
        """刷新类别复选框"""
        # 清空原有
        for cb in self.density_checkboxes:
            self.class_layout.removeWidget(cb)
            cb.deleteLater()
        self.density_checkboxes = []
        # 重新生成
        for cls in self.density_classes:
            cb = QCheckBox(cls)
            cb.setChecked(True)
            cb.stateChanged.connect(self.update_density_classes)
            self.class_layout.addWidget(cb)
            self.density_checkboxes.append(cb)

    def update_density_classes(self):
        """更新密度图类别"""
        self.density_classes = set()
        for cb in self.density_checkboxes:
            if cb.isChecked():
                self.density_classes.add(cb.text())

    def get_result(self):
        """获取对话框结果"""
        return self.density_classes
