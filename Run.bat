@echo off
cd /d %~dp0
python ClearData.py
start "" cmd /k "python YoloData.py" && start "" cmd /k "python Main.py"