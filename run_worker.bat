@echo off
cd /d %~dp0
py -m workers.trader_worker
pause