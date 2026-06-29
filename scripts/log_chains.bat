@echo off
REM Daily end-of-day option-chain snapshot for stock_research (builds IV history).
REM Scheduled via Windows Task Scheduler; see README. Output appended to data/chains/.
cd /d "C:\Users\joeyh\Documents\GitHub\stock_research"
set "PYTHONPATH=src"
".venv\Scripts\python.exe" -m stock_research.cli log-chains --quiet >> "data\chains\cronlog.txt" 2>&1
