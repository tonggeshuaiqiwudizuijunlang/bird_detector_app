@echo off
chcp 65001 > nul
echo 正在安装鸟类检测系统所需依赖...
echo =============================================

:: 检查是否以管理员身份运行
>nul 2>&1 "%SYSTEMROOT%\system32\cacls.exe" "%SYSTEMROOT%\system32\config\system" && (
    goto :install
) || (
    echo 请以管理员身份运行此脚本以安装依赖。
    echo 请右键点击此文件，选择"以管理员身份运行"。
    pause
    exit /b 1
)

:install
:: 安装依赖
echo 正在更新pip...
python -m pip install --upgrade pip

echo 正在安装依赖包...
python -m pip install -r requirements.txt

if %errorlevel% neq 0 (
    echo 依赖安装失败，请检查错误信息。
    echo 可能需要手动安装某些库：
    echo pip install PyQt5 numpy opencv-python matplotlib pandas ultralytics pyinstaller
    pause
    exit /b 1
)

echo =============================================
echo 依赖安装完成！现在可以运行或打包程序了。
echo =============================================
pause
