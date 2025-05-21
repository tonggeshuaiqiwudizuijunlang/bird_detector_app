"""
主应用程序模块
Creater Tz2H
"""

import csv
import gc
import os
from datetime import datetime

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtCore import QDateTime, Qt, QTimer
from PyQt5.QtGui import (
    QIcon,
    QImage,
    QPixmap,
)
from PyQt5.QtWidgets import (
    QAction,
    QComboBox,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMenu,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QStatusBar,
    QSystemTrayIcon,
    QVBoxLayout,
    QWidget,
)
from ui.components import MacStyleButton, MacStyleFrame
from ui.dialogs import DensityDialog, SettingsDialog
from utils.config_manager import save_config

from bird_detector_app.detector import ObjectDetector


class YoloVisualizationApp(QMainWindow):
    """YOLO可视化应用主窗口"""

    def __init__(self):
        """初始化主窗口"""
        super().__init__()
        self.setWindowTitle("鸟类检测系统")
        self.setGeometry(100, 100, 1440, 900)

        # 初始化属性
        self.all_classes = []
        self.selected_classes = set()
        self.density_classes = set()
        self.model_path = None
        self.is_detecting = False
        self.frame_count = 0
        self.fps = 0
        self.last_fps_update = QDateTime.currentDateTime()
        self.available_cameras = self.detect_cameras()
        self.selected_camera = None
        self.last_frame_time = QDateTime.currentDateTime()

        # 初始化检测器
        self.bird_detector = ObjectDetector("resources/models/yolo11m.pt")

        # 设置应用程序样式
        self.set_application_style()

        # 创建中央窗口部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # 创建菜单栏
        self.create_menu_bar()

        # 创建工具栏
        self.create_tool_bar()

        # 创建状态栏
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage("系统就绪")

        # 主布局 (垂直布局)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(20, 20, 20, 20)

        # 顶部按钮区域 (水平布局，包含设置和保存)
        self.create_top_buttons(main_layout)

        # 中间内容区域 (水平布局，包含左侧视频/计数和右侧密度图)
        content_layout = QHBoxLayout()
        content_layout.setSpacing(20)

        # 左侧布局（视频和计数）
        self.create_left_panel(content_layout)

        # 右侧布局（密度图和开始/停止）
        self.create_right_panel(content_layout)

        # 将内容布局添加到主布局下方
        main_layout.addLayout(content_layout, 1)

        # 初始化matplotlib FigureCanvas并添加到self.density_chart_placeholder区域
        self.init_matplotlib_canvas()

        # 初始化密度图类别 (默认与识别类别一致)
        self.density_classes = set(self.all_classes)
        self.bird_detector.density_classes = set(self.all_classes)

        # 初始化视频捕获
        self.cap = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(1)  # 设置为1毫秒，让系统尽可能快地更新

        # 存储识别数据 (密度图只用时间戳和总数)
        self.recognition_data = []  # [(timestamp, total_count, {class_name: count}), ...]

        # 系统托盘图标
        self.create_tray_icon()

        # 尝试从config.txt加载配置
        self.load_config()

    def set_application_style(self):
        """设置应用程序样式"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1a1a1a;
            }
            QWidget {
                background-color: transparent;
                color: #ffffff;
            }
            QLabel {
                color: #ffffff;
                font-size: 13px;
                font-weight: 400;
            }
            QMenuBar {
                background-color: rgba(26, 26, 26, 0.8);
                color: white;
                border-bottom: 1px solid rgba(255, 255, 255, 0.1);
            }
            QMenuBar::item:selected {
                background-color: rgba(255, 255, 255, 0.1);
                border-radius: 4px;
            }
            QMenu {
                background-color: rgba(26, 26, 26, 0.95);
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 8px;
                padding: 5px;
            }
            QMenu::item:selected {
                background-color: rgba(255, 255, 255, 0.1);
                border-radius: 4px;
            }
            QStatusBar {
                background-color: rgba(26, 26, 26, 0.8);
                color: #ffffff;
                border-top: 1px solid rgba(255, 255, 255, 0.1);
            }
            QComboBox {
                background-color: rgba(255, 255, 255, 0.1);
                color: white;
                border: 1px solid rgba(255, 255, 255, 0.2);
                border-radius: 6px;
                padding: 5px 10px;
                min-height: 25px;
            }
            QComboBox:hover {
                background-color: rgba(255, 255, 255, 0.15);
            }
            QComboBox::drop-down {
                border: none;
                width: 20px;
            }
            QComboBox::down-arrow {
                image: url(resources/icons/down_arrow.png);
                width: 12px;
                height: 12px;
            }
            QProgressBar {
                background-color: rgba(255, 255, 255, 0.1);
                border: none;
                border-radius: 3px;
                text-align: center;
                color: white;
            }
            QProgressBar::chunk {
                background-color: rgba(0, 122, 255, 0.8);
                border-radius: 3px;
            }
            QToolBar {
                background-color: rgba(26, 26, 26, 0.8);
                border: none;
                spacing: 5px;
            }
            QToolBar::separator {
                width: 1px;
                background-color: rgba(255, 255, 255, 0.1);
                margin: 5px;
            }
        """)

    def create_top_buttons(self, main_layout):
        """创建顶部按钮区域"""
        top_buttons_layout = QHBoxLayout()
        top_buttons_layout.setSpacing(10)

        # 设置按钮
        self.settings_btn = MacStyleButton("设置")
        self.settings_btn.setFixedWidth(100)
        self.settings_btn.clicked.connect(self.show_settings_dialog)
        top_buttons_layout.addWidget(self.settings_btn, alignment=Qt.AlignLeft)

        top_buttons_layout.addStretch()

        # 保存CSV按钮
        self.save_csv_button = MacStyleButton("保存数据为 CSV")
        self.save_csv_button.setIcon(
            self.style().standardIcon(self.style().SP_DialogSaveButton)
        )
        self.save_csv_button.clicked.connect(self.save_data_to_csv)
        top_buttons_layout.addWidget(self.save_csv_button, alignment=Qt.AlignRight)

        main_layout.addLayout(top_buttons_layout)

    def create_left_panel(self, content_layout):
        """创建左侧面板"""
        left_frame = MacStyleFrame()
        left_layout = QVBoxLayout(left_frame)
        left_layout.setSpacing(15)

        # 视频控制区域
        video_control_frame = MacStyleFrame()
        video_control_layout = QHBoxLayout(video_control_frame)

        # 视频源选择
        self.source_combo = QComboBox()
        self.source_combo.addItems(["摄像头", "视频文件"])
        video_control_layout.addWidget(QLabel("视频源:"))
        video_control_layout.addWidget(self.source_combo)

        # 分辨率选择
        self.resolution_combo = QComboBox()
        self.resolution_combo.addItems(["640x480", "1280x720", "1920x1080"])
        video_control_layout.addWidget(QLabel("分辨率:"))
        video_control_layout.addWidget(self.resolution_combo)

        left_layout.addWidget(video_control_frame)

        # 视频显示区域
        self.video_label = QLabel("等待视频流...")
        self.video_label.setMinimumSize(640, 640)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("""
            QLabel {
                background-color: rgba(0, 0, 0, 0.5);
                border: 1px solid rgba(255, 255, 255, 0.2);
                border-radius: 10px;
                padding: 10px;
            }
        """)
        left_layout.addWidget(self.video_label)

        # 识别信息区域
        info_frame = MacStyleFrame()
        info_layout = QHBoxLayout(info_frame)

        # 识别计数标签
        self.count_label = QLabel("识别到的鸟类数量: 0")
        self.count_label.setStyleSheet("""
            QLabel {
                font-size: 14px;
                font-weight: 500;
                color: #ffffff;
            }
        """)
        info_layout.addWidget(self.count_label)

        # FPS显示
        self.fps_label = QLabel("FPS: 0")
        self.fps_label.setStyleSheet("""
            QLabel {
                font-size: 14px;
                font-weight: 500;
                color: #ffffff;
            }
        """)
        info_layout.addWidget(self.fps_label)

        left_layout.addWidget(info_frame)

        # 将左侧布局添加到内容布局
        content_layout.addWidget(left_frame, 2)

    def create_right_panel(self, content_layout):
        """创建右侧面板"""
        right_frame = MacStyleFrame()
        right_layout = QVBoxLayout(right_frame)
        right_layout.setSpacing(15)

        # 数量密度分布图区域 (确保在右上角)
        self.density_chart_placeholder = QLabel("数量密度分布图")
        self.density_chart_placeholder.setMinimumSize(400, 300)
        self.density_chart_placeholder.setAlignment(Qt.AlignCenter)
        self.density_chart_placeholder.setStyleSheet("""
            QLabel {
                background-color: rgba(0, 0, 0, 0.5);
                border: 1px solid rgba(255, 255, 255, 0.2);
                border-radius: 10px;
                padding: 10px;
            }
        """)
        right_layout.addWidget(self.density_chart_placeholder)

        # 控制按钮区域
        control_frame = MacStyleFrame()
        control_layout = QVBoxLayout(control_frame)
        control_layout.setSpacing(10)

        # 开始/停止按钮
        self.start_stop_button = MacStyleButton("开始检测")
        self.start_stop_button.setIcon(
            self.style().standardIcon(self.style().SP_MediaPlay)
        )
        self.start_stop_button.clicked.connect(self.toggle_detection)
        control_layout.addWidget(self.start_stop_button)

        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        control_layout.addWidget(self.progress_bar)

        right_layout.addWidget(control_frame)

        # 将右侧布局添加到内容布局
        content_layout.addWidget(right_frame, 1)

    def init_matplotlib_canvas(self):
        """初始化matplotlib画布"""
        self.fig, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.fig)
        # 移除旧的占位符布局
        old_layout = self.density_chart_placeholder.layout()
        if old_layout:
            while old_layout.count():
                item = old_layout.takeAt(0)
                widget = item.widget()
                if widget:
                    widget.deleteLater()
            old_layout.deleteLater()
        new_layout = QVBoxLayout(self.density_chart_placeholder)
        new_layout.addWidget(self.canvas)
        new_layout.setContentsMargins(0, 0, 0, 0)
        new_layout.setSpacing(0)

    def load_config(self):
        """加载配置"""
        config_file = "config.txt"
        if os.path.exists(config_file):
            try:
                with open(config_file, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith("model="):
                            model_path = line.split("=", 1)[1]
                            if os.path.exists(model_path):
                                self.load_model_and_classes(model_path)
                                self.statusBar.showMessage(
                                    f"已加载模型: {os.path.basename(model_path)}"
                                )
                            else:
                                self.statusBar.showMessage("配置中指定的模型文件不存在")
                        elif line.startswith("classes="):
                            classes_str = line.split("=", 1)[1]
                            if classes_str:
                                self.selected_classes = set(classes_str.split(","))
                                self.bird_detector.selected_classes = (
                                    self.selected_classes
                                )
                                # 如果密度图类别未设置，默认与识别类别一致
                                if (
                                    not hasattr(self, "density_classes")
                                    or not self.density_classes
                                ):
                                    self.density_classes = set(self.selected_classes)
                                    self.bird_detector.density_classes = set(
                                        self.selected_classes
                                    )
                            else:
                                self.selected_classes = set()
                                self.bird_detector.selected_classes = set()
                                self.density_classes = set()
                                self.bird_detector.density_classes = set()
            except Exception as e:
                self.statusBar.showMessage(f"读取config.txt失败: {e}")
        else:
            self.statusBar.showMessage("未找到config.txt文件，请进行设置")

    def create_menu_bar(self):
        """创建菜单栏"""
        menubar = self.menuBar()

        # 文件菜单
        file_menu = menubar.addMenu("文件")

        open_action = QAction("打开视频", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self.open_video)
        file_menu.addAction(open_action)

        save_action = QAction("保存数据", self)
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self.save_data_to_csv)
        file_menu.addAction(save_action)

        file_menu.addSeparator()

        exit_action = QAction("退出", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # 视图菜单
        view_menu = menubar.addMenu("视图")

        fullscreen_action = QAction("全屏", self)
        fullscreen_action.setShortcut("F11")
        fullscreen_action.triggered.connect(self.toggle_fullscreen)
        view_menu.addAction(fullscreen_action)

        # 帮助菜单
        help_menu = menubar.addMenu("帮助")

        about_action = QAction("关于", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)

    def create_tool_bar(self):
        """创建工具栏"""
        # 工具栏已不再包含设置和保存按钮，故留空或添加其他常用操作
        pass

    def create_tray_icon(self):
        """创建系统托盘图标"""
        self.tray_icon = QSystemTrayIcon(self)
        # 检查self.style()是否为None，或使用默认图标路径
        icon = (
            self.style().standardIcon(self.style().SP_ComputerIcon)
            if self.style()
            else QIcon("resources/icons/app_icon.png")
        )
        if icon.isNull():
            icon = QIcon("resources/icons/app_icon.png")  # 尝试使用项目内相对路径
        self.tray_icon.setIcon(icon)

        # 创建托盘菜单
        tray_menu = QMenu()
        show_action = tray_menu.addAction("显示")
        show_action.triggered.connect(self.show)
        quit_action = tray_menu.addAction("退出")
        quit_action.triggered.connect(self.close)

        self.tray_icon.setContextMenu(tray_menu)
        self.tray_icon.show()

    def show_settings_dialog(self):
        """显示设置对话框"""
        dlg = QDialog(self)
        dlg.setWindowTitle("设置")
        dlg.resize(300, 150)
        layout = QVBoxLayout(dlg)

        model_btn = QPushButton("模型设置")
        density_btn = QPushButton("密度图设置")

        layout.addWidget(model_btn)
        layout.addWidget(density_btn)

        def on_model():
            # 这里只处理模型选择和识别类别
            sdlg = SettingsDialog(
                self, self.model_path, self.all_classes, self.selected_classes
            )
            if sdlg.exec_():
                model_path, selected_classes = sdlg.get_result()
                self.load_model_and_classes(model_path)
                self.selected_classes = selected_classes
                self.bird_detector.selected_classes = self.selected_classes
                # 保存到config.txt
                save_config(model_path, selected_classes)
                # 如果密度图类别未设置，默认与识别类别一致
                if not hasattr(self, "density_classes") or not self.density_classes:
                    self.density_classes = set(selected_classes)
                    self.bird_detector.density_classes = set(selected_classes)

        def on_density():
            # 这里只处理密度图类别选择
            # DensityDialog只需要当前识别类别列表供选择
            ddialog = DensityDialog(self, density_classes=list(self.selected_classes))
            if ddialog.exec_():
                self.density_classes = ddialog.get_result()
                self.bird_detector.density_classes = self.density_classes

        model_btn.clicked.connect(on_model)
        density_btn.clicked.connect(on_density)

        dlg.exec_()

    def toggle_detection(self):
        """切换检测状态"""
        self.is_detecting = not self.is_detecting
        if self.is_detecting:
            self.start_stop_button.setText("停止检测")
            self.start_stop_button.setIcon(
                self.style().standardIcon(self.style().SP_MediaStop)
            )
            self.statusBar.showMessage("检测中...")
        else:
            self.start_stop_button.setText("开始检测")
            self.start_stop_button.setIcon(
                self.style().standardIcon(self.style().SP_MediaPlay)
            )
            self.statusBar.showMessage("检测已停止")
            # 清空当前检测信息
            if hasattr(self.bird_detector, "current_detection_info"):
                self.bird_detector.current_detection_info = []
            self.bird_detector.total_objects = 0
            self.count_label.setText("识别到的鸟类数量: 0")

    def open_video(self):
        """打开视频文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "打开视频文件", "", "视频文件 (*.mp4 *.avi *.mkv)"
        )
        if file_path:
            # 如果cap已打开，先释放
            if self.cap and self.cap.isOpened():
                self.cap.release()
            self.cap = cv2.VideoCapture(file_path)
            if not self.cap.isOpened():
                self.statusBar.showMessage(
                    f"无法打开视频文件: {os.path.basename(file_path)}"
                )
                self.cap = None
            else:
                self.statusBar.showMessage(f"已打开视频: {os.path.basename(file_path)}")
                self.is_detecting = False  # 打开视频后停止检测
                self.start_stop_button.setText("开始检测")
                self.start_stop_button.setIcon(
                    self.style().standardIcon(self.style().SP_MediaPlay)
                )

    def toggle_fullscreen(self):
        """切换全屏状态"""
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()

    def show_about(self):
        """显示关于对话框"""
        QMessageBox.about(
            self,
            "关于",
            "YOLO智能识别分析系统\n" "版本: 1.0.0\n" "© 2025 版权所有:睿翼智控",
        )

    def detect_cameras(self):
        """检测系统中可用的摄像头"""
        available_cameras = []
        for i in range(10):  # 检查前10个摄像头索引
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    available_cameras.append(i)
                cap.release()
        return available_cameras

    def show_camera_selection_dialog(self):
        """显示摄像头选择对话框"""
        if not self.available_cameras:
            QMessageBox.warning(self, "警告", "未检测到可用的摄像头！")
            return False

        dialog = QDialog(self)
        dialog.setWindowTitle("选择摄像头")
        layout = QVBoxLayout(dialog)

        # 添加说明标签
        layout.addWidget(QLabel("请选择要使用的摄像头："))

        # 创建摄像头选择下拉框
        camera_combo = QComboBox()
        for camera_id in self.available_cameras:
            camera_combo.addItem(f"摄像头 {camera_id}", camera_id)
        layout.addWidget(camera_combo)

        # 添加按钮
        button_layout = QHBoxLayout()
        ok_button = QPushButton("确定")
        cancel_button = QPushButton("取消")
        button_layout.addWidget(ok_button)
        button_layout.addWidget(cancel_button)
        layout.addLayout(button_layout)

        # 连接按钮信号
        ok_button.clicked.connect(dialog.accept)
        cancel_button.clicked.connect(dialog.reject)

        # 显示对话框
        if dialog.exec_() == QDialog.Accepted:
            self.selected_camera = camera_combo.currentData()
            return True
        return False

    def update_frame(self):
        """更新视频帧并进行检测"""
        # 计算实际FPS
        current_time = QDateTime.currentDateTime()
        elapsed = self.last_frame_time.msecsTo(current_time)
        if elapsed > 0:  # 避免除以零
            current_fps = 1000 / elapsed
            self.fps = (self.fps * 0.9) + (current_fps * 0.1)  # 平滑FPS显示
        self.last_frame_time = current_time

        if not self.is_detecting:
            # 如果停止检测，继续读取帧并显示，但不进行推理和统计
            if self.cap is None:
                if self.selected_camera is None:
                    # 如果没有选中的摄像头，弹出选择对话框
                    if not self.show_camera_selection_dialog():
                        return
                self.cap = cv2.VideoCapture(self.selected_camera)
                # 设置摄像头分辨率为640x640
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
                # 设置摄像头缓冲区大小
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                if not self.cap.isOpened():
                    # 显示摄像头未打开的占位符
                    black_image = np.zeros((640, 640, 3), dtype=np.uint8)
                    no_camera_icon_path = "resources/icons/no_camera.png"
                    if os.path.exists(no_camera_icon_path):
                        icon = cv2.imread(no_camera_icon_path, cv2.IMREAD_UNCHANGED)
                        if icon is not None:
                            # 调整图标大小并叠加到黑色背景
                            icon_height, icon_width = icon.shape[:2]
                            scale = min(400 / icon_width, 300 / icon_height)
                            resized_icon = cv2.resize(
                                icon,
                                (int(icon_width * scale), int(icon_height * scale)),
                            )
                            h, w = black_image.shape[:2]
                            ih, iw = resized_icon.shape[:2]
                            x = (w - iw) // 2
                            y = (h - ih) // 2
                            if resized_icon.shape[2] == 4:
                                alpha_s = resized_icon[:, :, 3] / 255.0
                                alpha_l = 1.0 - alpha_s
                                for c in range(0, 3):
                                    black_image[y : y + ih, x : x + iw, c] = (
                                        alpha_s * resized_icon[:, :, c]
                                        + alpha_l
                                        * black_image[y : y + ih, x : x + iw, c]
                                    )
                            else:
                                black_image[y : y + ih, x : x + iw] = resized_icon[
                                    :, :, :3
                                ]
                    processed_frame = black_image
                    self.count_label.setText("识别到的鸟类数量: 0")
                    self.fps_label.setText(f"FPS: {self.fps:.1f}")
                    h, w, ch = processed_frame.shape
                    bytes_per_line = ch * w
                    qt_image = QImage(
                        processed_frame.data, w, h, bytes_per_line, QImage.Format_RGB888
                    ).rgbSwapped()
                    pixmap = QPixmap.fromImage(qt_image)
                    self.video_label.setPixmap(
                        pixmap.scaled(
                            self.video_label.width(),
                            self.video_label.height(),
                            Qt.KeepAspectRatio,
                        )
                    )
                    return

            ret, frame = self.cap.read()
            if not ret:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.cap.read()
                if not ret:
                    self.statusBar.showMessage("视频播放完毕或无法读取帧")
                    return

            # 显示非检测状态下的画面
            processed_frame = frame
            self.count_label.setText("识别到的鸟类数量: 0")
            self.fps_label.setText(f"FPS: {self.fps:.1f}")
            h, w, ch = processed_frame.shape
            bytes_per_line = ch * w
            qt_image = QImage(
                processed_frame.data, w, h, bytes_per_line, QImage.Format_RGB888
            ).rgbSwapped()
            pixmap = QPixmap.fromImage(qt_image)
            self.video_label.setPixmap(
                pixmap.scaled(
                    self.video_label.width(),
                    self.video_label.height(),
                    Qt.KeepAspectRatio,
                )
            )
            return

        if self.cap is None:
            if self.selected_camera is None:
                if not self.show_camera_selection_dialog():
                    self.is_detecting = False
                    self.start_stop_button.setText("开始检测")
                    self.start_stop_button.setIcon(
                        self.style().standardIcon(self.style().SP_MediaPlay)
                    )
                    return
            self.cap = cv2.VideoCapture(self.selected_camera)
            # 设置摄像头分辨率为640x640
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
            # 设置摄像头缓冲区大小
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if not self.cap.isOpened():
                self.statusBar.showMessage("摄像头无法打开或不可用")
                self.is_detecting = False
                self.start_stop_button.setText("开始检测")
                self.start_stop_button.setIcon(
                    self.style().standardIcon(self.style().SP_MediaPlay)
                )
                return

        ret, frame = self.cap.read()
        if not ret:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()
            if not ret:
                self.statusBar.showMessage("视频播放完毕或无法读取帧")
                self.is_detecting = False
                self.start_stop_button.setText("开始检测")
                self.start_stop_button.setIcon(
                    self.style().standardIcon(self.style().SP_MediaPlay)
                )
                return

        # 处理帧
        processed_frame = self.bird_detector.process_frame(frame)

        # 更新计数标签
        self.count_label.setText(
            f"识别到的鸟类数量: {self.bird_detector.total_objects}"
        )

        # 记录数量密度数据
        now_str = datetime.now().strftime("%H:%M:%S")
        current_frame_class_counts = {}
        if hasattr(self.bird_detector, "current_detection_info"):
            for det_info in self.bird_detector.current_detection_info:
                class_name = det_info["class"]
                if class_name in self.density_classes:
                    current_frame_class_counts[class_name] = (
                        current_frame_class_counts.get(class_name, 0) + 1
                    )

        total_objects_for_density = sum(current_frame_class_counts.values())

        if total_objects_for_density > 0:
            self.recognition_data.append(
                (now_str, total_objects_for_density, current_frame_class_counts)
            )
            if len(self.recognition_data) > 100:
                self.recognition_data.pop(0)

        # 更新视频显示
        h, w, ch = processed_frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(
            processed_frame.data, w, h, bytes_per_line, QImage.Format_RGB888
        ).rgbSwapped()
        pixmap = QPixmap.fromImage(qt_image)
        self.video_label.setPixmap(
            pixmap.scaled(
                self.video_label.width(), self.video_label.height(), Qt.KeepAspectRatio
            )
        )

        # 更新图表
        self.update_density_chart()

    def update_density_chart(self):
        """更新密度图表"""
        if not self.recognition_data or not getattr(self, "density_classes", None):
            self.ax.clear()
            self.ax.set_title("数量密度分布（暂无数据）")
            self.canvas.draw()
            return

        # 统计每个类别的时间序列
        from collections import defaultdict

        class_time_count = defaultdict(list)
        timestamps = [item[0] for item in self.recognition_data]
        # 统计每个类别在每个时间点的数量
        for idx, (t, _, frame_classes) in enumerate(self.recognition_data):
            for cls in self.density_classes:
                class_time_count[cls].append(frame_classes.get(cls, 0))

        self.ax.clear()
        # 兼容新版matplotlib的colormap获取方式
        if hasattr(matplotlib, "colormaps"):
            # 只为实际需要绘制的类别分配颜色
            valid_classes = [
                cls for cls in self.density_classes if any(class_time_count[cls])
            ]
            if not valid_classes:
                self.ax.set_title("数量密度分布（暂无数据）")
                self.canvas.draw()
                return
            color_map = matplotlib.colormaps.get_cmap("tab10").resampled(
                max(1, len(valid_classes))
            )
            for i, cls in enumerate(valid_classes):
                y = class_time_count[cls]
                color = color_map(i)
                self.ax.plot(
                    timestamps,
                    y,
                    label=cls,
                    linewidth=2.5,
                    marker="o",
                    markersize=7,
                    color=color,
                )
        else:
            import matplotlib.cm as cm

            # 只为实际需要绘制的类别分配颜色
            valid_classes = [
                cls for cls in self.density_classes if any(class_time_count[cls])
            ]
            if not valid_classes:
                self.ax.set_title("数量密度分布（暂无数据）")
                self.canvas.draw()
                return
            color_map = cm.get_cmap("tab10", max(1, len(valid_classes)))
            for i, cls in enumerate(valid_classes):
                y = class_time_count[cls]
                color = (
                    color_map(i)
                    if hasattr(color_map, "__call__")
                    else color_map.colors[i]
                )
                self.ax.plot(
                    timestamps,
                    y,
                    label=cls,
                    linewidth=2.5,
                    marker="o",
                    markersize=7,
                    color=color,
                )

        self.ax.set_xlabel("时间", fontsize=12)
        self.ax.set_ylabel("数量", fontsize=12)
        self.ax.set_title("数量密度分布", fontsize=14, fontweight="bold")
        self.ax.grid(True, linestyle="--", alpha=0.4)
        # x轴最多显示10个标签
        step = max(1, len(timestamps) // 10)
        self.ax.set_xticks(timestamps[::step])
        self.ax.tick_params(axis="x", labelrotation=45)
        self.ax.legend(
            fontsize=12, loc="upper left", frameon=True, fancybox=True, shadow=True
        )
        self.canvas.draw()

    def save_data_to_csv(self):
        """保存检测数据到CSV文件"""
        if (
            not self.bird_detector
            or not hasattr(self.bird_detector, "current_detection_info")
            or not self.bird_detector.current_detection_info
        ):
            self.statusBar.showMessage("没有检测数据可保存")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存数据", "", "CSV 文件 (*.csv)"
        )
        if file_path:
            try:
                # 写入所有检测到的目标，不论是否勾选显示
                # 写入表头
                with open(file_path, "w", newline="", encoding="utf-8") as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(["时间戳", "类别", "总数量"])
                    # 写入每一帧的所有检测信息
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    total_objects = len(self.bird_detector.current_detection_info)
                    for info in self.bird_detector.current_detection_info:
                        writer.writerow([timestamp, info["class"], total_objects])
                self.statusBar.showMessage(f"数据已保存到 {file_path}")
            except Exception as e:
                self.statusBar.showMessage(f"保存文件失败: {e}")

    def closeEvent(self, event):
        """关闭窗口事件处理"""
        reply = QMessageBox.question(
            self,
            "确认退出",
            "确定要退出程序吗？",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )

        if reply == QMessageBox.Yes:
            # 停止摄像头
            if self.cap and self.cap.isOpened():
                self.cap.release()
            self.timer.stop()
            cv2.destroyAllWindows()
            # 生成趋势图（使用保存的CSV文件，如果存在）
            try:
                # 调用ObjectDetector的plot_trends方法
                if hasattr(self, "bird_detector") and self.bird_detector:
                    self.bird_detector.plot_trends()
            except Exception as e:
                print(f"生成趋势图时出错: {e}")
            event.accept()
        else:
            event.ignore()

    def load_model_and_classes(self, model_path):
        """加载模型和类别"""
        # 主动释放旧模型
        if hasattr(self, "bird_detector") and self.bird_detector is not None:
            del self.bird_detector
            gc.collect()
        try:
            self.model_path = model_path
            self.bird_detector = ObjectDetector(model_path)
            self.all_classes = list(self.bird_detector.model.names.values())
            # 初始时，识别类别和密度图类别都等于模型的全部类别
            if not hasattr(self, "selected_classes") or not self.selected_classes:
                self.selected_classes = set(self.all_classes)
            if not hasattr(self, "density_classes") or not self.density_classes:
                self.density_classes = set(self.all_classes)

            self.bird_detector.selected_classes = self.selected_classes
            self.bird_detector.density_classes = self.density_classes

            self.statusBar.showMessage(f"成功加载模型: {os.path.basename(model_path)}")

        except Exception as e:
            self.statusBar.showMessage(f"加载模型失败: {e}")
            # 如果加载失败，清空类别和模型路径
            self.model_path = None
            self.all_classes = []
            self.selected_classes = set()
            self.density_classes = set()
            self.bird_detector = None  # 清空检测器对象
