@echo off
chcp 65001 >nul
setlocal
set PYTHONUNBUFFERED=1
set PYTHONIOENCODING=utf-8

python "%~dp0..\src\mcp\media_mcp.py" %*
