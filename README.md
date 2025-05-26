# 鸟类检测系统

基于 YOLO 的智能鸟类检测系统，使用 PyQt5 构建用户界面。

## 功能特点

- 实时摄像头检测
- 视频文件检测
- 多类别检测与过滤
- 检测数据统计与可视化
- 数据保存与导出

## 系统要求

- Python 3.8+
- OpenCV
- PyQt5
- Ultralytics YOLO
- PyInstaller (用于打包)
- 其他依赖见 requirements.txt

## 安装步骤

1. 安装依赖:

   ```bash
   pip install -r requirements.txt
   ```

   或者运行依赖安装脚本 (Windows):

   ```bash
   requirementInstallation.bat

   ```

2. 启动程序:

   ```bash
   python main.py
   ```

## 打包应用程序 (生成 EXE)

1. **确保 PyInstaller 已安装**: 如果未包含在 `requirements.txt` 中或未安装，请先安装：

   ```bash
   pip install pyinstaller
   ```

2. **打包**:
   在项目根目录下打开终端或命令行，然后运行：

   ```bash
   pyinstaller BirdDetectorApp.spec
   ```

   脚本会自动处理依赖、包含必要的资源文件，并在 `dist` 文件夹下生成 `BirdDetectorApp.exe`。

## 项目结构

```
bird_detector_app/
├── bird_detector_app/     # 主程序包
│   ├── __init__.py
│   ├── app.py             # 主应用类
│   └── detector.py        # 检测器类
├── resources/             # 资源文件
│   ├── icons/             # 图标资源
│   └── models/            # 模型文件
├── ui/                    # UI组件
│   ├── __init__.py
│   ├── components.py      # 自定义控件
│   └── dialogs.py         # 对话框
├── utils/                 # 实用工具
│   ├── __init__.py
│   └── config_manager.py  # 配置管理
├── main.py                # 程序入口
├── build_exe.py           # PyInstaller 打包脚本
├── requirements.txt       # 依赖列表
├── requirementInstallation.bat # Windows 依赖安装脚本
└── README.md              # 项目说明文件
```

## 使用方法

1. 启动程序
2. 使用设置菜单选择模型和需要检测的类别
3. 选择视频源（摄像头或视频文件）
4. 点击"开始检测"按钮进行检测
5. 检测结果将显示在界面上，同时可保存为 CSV 文件

## 许可证

© 2025 版权所有 Tz2H
