@echo off
cd /d %~dp0
python -m workers.trader_worker
