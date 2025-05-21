"""
鸟类检测系统 - 主程序入口
Creater Tz2H
"""

import sys

from PyQt5.QtWidgets import QApplication
from utils.config_manager import load_initial_config

from bird_detector_app.app import YoloVisualizationApp


def main():
    """主程序入口函数"""
    # 检查是否存在config.txt并加载配置
    initial_config = load_initial_config()

    app = QApplication(sys.argv)

    # 确保在 QApplication 创建后初始化主窗口
    main_window = YoloVisualizationApp()

    # 在窗口显示前加载模型和类别
    main_window.load_model_and_classes(initial_config["model_path"])
    if initial_config["selected_classes"]:
        main_window.selected_classes = initial_config["selected_classes"]
        main_window.bird_detector.selected_classes = initial_config["selected_classes"]
    if initial_config["density_classes"]:
        main_window.density_classes = initial_config["density_classes"]
        main_window.bird_detector.density_classes = initial_config["density_classes"]

    main_window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
