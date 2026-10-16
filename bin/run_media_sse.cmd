@echo off
chcp 65001 >nul
title Media MCP - Unified Server (SSE :7865)
setlocal
cd /d "%~dp0.."
set PYTHONUNBUFFERED=1
set PYTHONIOENCODING=utf-8

python src\mcp\media_mcp.py --sse %*
pause
