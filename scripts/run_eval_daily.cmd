@echo off
REM Collect one day's worth of the eval sweep, staying inside the free-tier
REM allowance (OpenRouter free models: 50 requests/day).
REM
REM The runner resumes from evals/results.jsonl, retries anything that failed
REM previously, and stops on its own when the budget or the quota runs out.
REM Registered as a Scheduled Task; see scripts\schedule_eval.ps1.

setlocal
set ROOT=%~dp0..
cd /d "%ROOT%"

if not exist "logs" mkdir "logs"

echo. >> logs\eval.log
echo ================================================== >> logs\eval.log
echo run started %DATE% %TIME% >> logs\eval.log
echo ================================================== >> logs\eval.log

".venv\Scripts\python.exe" -W ignore -m evals.run_eval --max-calls 45 >> logs\eval.log 2>&1

echo run finished %DATE% %TIME% with exit code %ERRORLEVEL% >> logs\eval.log
endlocal
