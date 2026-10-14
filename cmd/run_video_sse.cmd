@echo off
chcp 65001 >nul
title Media MCP - Video Server (SSE :7866)
setlocal
cd /d "%~dp0.."
set PYTHONUNBUFFERED=1
set PYTHONIOENCODING=utf-8

set "CFG_PARAM="
if exist "G:\Git\Gemini\media_mcp\config.json" (
    set CFG_PARAM=--config "G:\Git\Gemini\media_mcp\config.json"
)

python src\mcp\video_mcp.py --sse %CFG_PARAM% %*
pause
