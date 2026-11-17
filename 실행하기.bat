@echo off
cd /d "%~dp0"
where python >nul 2>nul || (echo Python is not installed. & pause & exit /b 1)
python server.py
if errorlevel 1 pause
