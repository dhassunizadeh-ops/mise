@echo off
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
.venv\Scripts\python.exe -m uvicorn backend.main:app --reload --host 0.0.0.0 --port 3001
