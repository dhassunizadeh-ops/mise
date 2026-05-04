@echo off
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
.venv\Scripts\streamlit.exe run frontend/app.py --server.port 8501
