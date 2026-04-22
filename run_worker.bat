@echo off
setlocal
cd /d %~dp0

set "PYTHON_EXE=%TRADER_PREMIUM_MAX_PYTHON%"
if not defined PYTHON_EXE set "PYTHON_EXE=C:\Users\User\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
if not exist "%PYTHON_EXE%" set "PYTHON_EXE=python"

"%PYTHON_EXE%" -m workers.trader_worker
