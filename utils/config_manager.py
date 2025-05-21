"""
配置管理工具模块
Creater Tz2H
"""

import os


def load_initial_config():
    """加载初始配置"""
    # 默认配置
    config = {
        "model_path": "resources/models/yolo11m.pt",
        "selected_classes": set(),
        "density_classes": set(),
    }

    # 尝试从config.txt加载配置
    config_file = "config.txt"
    if os.path.exists(config_file):
        try:
            with open(config_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("model="):
                        config["model_path"] = line.split("=", 1)[1]
                    elif line.startswith("classes="):
                        classes_str = line.split("=", 1)[1]
                        if classes_str:
                            config["selected_classes"] = set(classes_str.split(","))
                            # 如果config有识别类别，密度图默认与识别类别一致
                            config["density_classes"] = set(config["selected_classes"])
        except Exception as e:
            print(f"读取config.txt失败: {e}")

    return config


def save_config(model_path, selected_classes, density_classes=None):
    """保存配置到文件"""
    config_file = "config.txt"
    try:
        with open(config_file, "w", encoding="utf-8") as f:
            f.write(f"model={model_path}\n")
            f.write("classes=" + ",".join(selected_classes) + "\n")
            if density_classes:
                f.write("density=" + ",".join(density_classes) + "\n")
        return True
    except Exception as e:
        print(f"保存config.txt失败: {e}")
        return False
