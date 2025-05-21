import subprocess
import os
import shutil

# --- Configuration ---
APP_NAME = "BirdDetectorApp"
MAIN_SCRIPT = "main.py"
ICON_PATH = "resources/icons/favicon.ico"  # Relative path to the script location
DIST_PATH = "dist"  # PyInstaller output directory
BUILD_PATH = "build"  # PyInstaller build working directory
SPEC_FILE = f"{APP_NAME}.spec"

# Hidden modules to include
HIDDEN_IMPORTS = [
    "utils.config_manager",
    "bird_detector_app.app",
    "bird_detector_app.detector",
    "ui.components",
    "ui.dialogs",
    # May need to add if some parts of ultralytics or its dependencies (like torch) are not auto-detected
    # e.g.: "torch", "torchvision", "torchaudio"
    # "ultralytics" # Usually PyInstaller's ultralytics hook handles this
]

# Data files and folders to add
# Format: "source_path;destination_path_in_package" or "source_path:destination_path_in_package"
# os.pathsep is ';' on Windows, ':' on Linux/macOS
ADD_DATA = [
    f"resources{os.pathsep}resources",
    f"config.txt{os.pathsep}.",
]
# Check if custom_hooks.py exists, if so, add it to ADD_DATA
if os.path.exists("custom_hooks.py"):
    ADD_DATA.append(f"custom_hooks.py{os.pathsep}.")
else:
    print(
        "Warning: 'custom_hooks.py' not found, it will not be included in --add-data."
    )


# --- Build Function ---
def build_executable():
    """
    Builds the executable using PyInstaller.
    """
    print(f"Starting to build executable for '{APP_NAME}'...")

    # 1. Clean up old build files and directories
    print("Cleaning up old build files...")
    if os.path.exists(DIST_PATH):
        shutil.rmtree(DIST_PATH)
        print(f"Deleted '{DIST_PATH}' directory.")
    if os.path.exists(BUILD_PATH):
        shutil.rmtree(BUILD_PATH)
        print(f"Deleted '{BUILD_PATH}' directory.")
    if os.path.exists(SPEC_FILE):
        os.remove(SPEC_FILE)
        print(f"Deleted '{SPEC_FILE}' file.")

    # 2. Construct PyInstaller command
    command = [
        "pyinstaller",
        "--name",
        APP_NAME,
        "--onefile",
        "--windowed",  # For GUI applications; use --console or remove for command-line apps
    ]

    if os.path.exists(ICON_PATH):
        command.extend(["--icon", ICON_PATH])
    else:
        print(f"Warning: Icon file '{ICON_PATH}' not found, will not use an icon.")

    for hidden_import in HIDDEN_IMPORTS:
        command.extend(["--hidden-import", hidden_import])

    for data_entry in ADD_DATA:
        command.extend(["--add-data", data_entry])

    command.append(MAIN_SCRIPT)

    print("\nWill execute the following PyInstaller command:")
    print(f"  {' '.join(command)}\n")

    # 3. Run PyInstaller
    try:
        print("Running PyInstaller...")
        # Explicitly use utf-8 encoding on Windows to avoid potential console output issues
        process = subprocess.run(
            command, check=True, capture_output=True, text=True, encoding="utf-8"
        )
        print("\n--- PyInstaller Output ---")
        print(process.stdout)
        if process.stderr:
            print("\n--- PyInstaller Errors (if any) ---")
            print(process.stderr)
        print(
            f"\nBuild successful! Executable is located at '{os.path.join(DIST_PATH, APP_NAME)}.exe'"
        )
    except subprocess.CalledProcessError as e:
        print("\n--- PyInstaller Build Failed ---")
        print("Error message:")
        # Ensure error output is also decoded as utf-8 (if text=True and encoding='utf-8' didn't fully cover it)
        stderr_output = (
            e.stderr.decode("utf-8", errors="replace")
            if isinstance(e.stderr, bytes)
            else e.stderr
        )
        stdout_output = (
            e.stdout.decode("utf-8", errors="replace")
            if isinstance(e.stdout, bytes)
            else e.stdout
        )
        print(stderr_output)
        print("\nStandard Output (if any):")
        print(stdout_output)
        print(
            "\nPlease check the error messages and ensure all dependencies and paths are correct."
        )
    except FileNotFoundError:
        print("Error: 'pyinstaller' command not found.")
        print(
            "Please ensure PyInstaller is installed and in your system's PATH environment variable."
        )
        print("You can install it by running 'pip install pyinstaller'.")
    except Exception as e:
        print(f"\nAn unknown error occurred during the build process: {e}")


if __name__ == "__main__":
    # Ensure the script runs from the project root directory (or adjust paths accordingly)
    # project_root = os.path.dirname(os.path.abspath(__file__))
    # os.chdir(project_root) # Uncomment to change working directory if needed
    build_executable()
