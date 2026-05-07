@echo off
setlocal enableextensions enabledelayedexpansion

REM Ingest every PDF in backend\PDFs\newset into Supabase via backend\scraper.py
REM Run this from the repo root (recommended) or any directory.
REM If you double-click this file, the window will close on completion; this script
REM pauses at the end so you can see any errors.

set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%..") do set "REPO_ROOT=%%~fI"

set "PDF_DIR=%REPO_ROOT%\backend\PDFs\newset"
set "LOG_DIR=%REPO_ROOT%\backend\logs"
set "SOURCE_TYPE=SAMA"

if not exist "%LOG_DIR%" mkdir "%LOG_DIR%" >nul 2>&1

REM Build a filesystem-safe timestamp: YYYYMMDD-HHMMSS
for /f %%A in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd-HHmmss"') do set "TS=%%A"
REM Add a random suffix so multiple launches in the same second don't collide
set "RUN_ID=%TS%_%RANDOM%%RANDOM%"
set "LOG_FILE=%LOG_DIR%\newset_ingest_%RUN_ID%.log"
set "FAILED_FILE=%LOG_DIR%\newset_failed_%RUN_ID%.txt"
set "LOCK_FILE=%LOG_DIR%\newset_ingest_%RUN_ID%.lock"

if exist "%FAILED_FILE%" del /q "%FAILED_FILE%" >nul 2>&1

REM Best-effort lock: prevents accidental double-click concurrent runs
2>nul ( >>"%LOCK_FILE%" echo started %DATE% %TIME% ) || (
  echo ERROR: Could not create lock file: "%LOCK_FILE%"
  echo ERROR: Could not create lock file: "%LOCK_FILE%" >> "%LOG_FILE%"
  goto :END_FAIL
)

echo ================================================== > "%LOG_FILE%"
echo IOTA v3 - Newset PDF ingestion >> "%LOG_FILE%"
echo Start time : %DATE% %TIME% >> "%LOG_FILE%"
echo Repo root  : %REPO_ROOT% >> "%LOG_FILE%"
echo PDF folder : %PDF_DIR% >> "%LOG_FILE%"
echo source_type: %SOURCE_TYPE% >> "%LOG_FILE%"
echo Log file   : %LOG_FILE% >> "%LOG_FILE%"
echo ================================================== >> "%LOG_FILE%"
echo. >> "%LOG_FILE%"

if not exist "%PDF_DIR%" (
  echo ERROR: Folder not found: "%PDF_DIR%"
  echo ERROR: Folder not found: "%PDF_DIR%" >> "%LOG_FILE%"
  goto :END_FAIL
)

REM Ensure there is at least one PDF to process
dir /b /a-d "%PDF_DIR%\*.pdf" >nul 2>&1
if errorlevel 1 (
  echo ERROR: No PDFs found in: "%PDF_DIR%"
  echo ERROR: No PDFs found in: "%PDF_DIR%" >> "%LOG_FILE%"
  goto :END_FAIL
)

REM Basic sanity checks (logged)
where python >nul 2>&1
if errorlevel 1 (
  echo ERROR: python not found on PATH.
  echo ERROR: python not found on PATH. >> "%LOG_FILE%"
  goto :END_FAIL
)

set /a total=0
set /a ok=0
set /a fail=0

pushd "%REPO_ROOT%" >nul 2>&1

for %%F in ("%PDF_DIR%\*.pdf") do (
  set /a total+=1
  echo Processing [!total!]: %%~nxF
  echo -------------------------------------------------- >> "%LOG_FILE%"
  echo [!total!] FILE: %%~nxF >> "%LOG_FILE%"
  echo CMD : python backend\scraper.py --file "%%~fF" --name "%%~nF" --source "%SOURCE_TYPE%" >> "%LOG_FILE%"
  echo -------------------------------------------------- >> "%LOG_FILE%"

  python backend\scraper.py --file "%%~fF" --name "%%~nF" --source "%SOURCE_TYPE%" >> "%LOG_FILE%" 2>&1
  set "EXITCODE=!errorlevel!"

  if not "!EXITCODE!"=="0" (
    set /a fail+=1
    >> "%FAILED_FILE%" echo - %%~nxF exit=!EXITCODE!
    echo RESULT: FAIL exit=!EXITCODE! >> "%LOG_FILE%"
    echo   -> FAIL exit=!EXITCODE!
  ) else (
    set /a ok+=1
    echo RESULT: OK >> "%LOG_FILE%"
    echo   -> OK
  )
  echo. >> "%LOG_FILE%"
)

popd >nul 2>&1

echo ================================================== >> "%LOG_FILE%"
echo INGESTION SUMMARY >> "%LOG_FILE%"
echo ================================================== >> "%LOG_FILE%"
echo Total PDFs    : %total% >> "%LOG_FILE%"
echo Succeeded     : %ok% >> "%LOG_FILE%"
echo Failed        : %fail% >> "%LOG_FILE%"
echo End time      : %DATE% %TIME% >> "%LOG_FILE%"
echo ================================================== >> "%LOG_FILE%"

if not "%fail%"=="0" (
  echo. >> "%LOG_FILE%"
  echo Failed files: >> "%LOG_FILE%"
  if exist "%FAILED_FILE%" (
    type "%FAILED_FILE%" >> "%LOG_FILE%"
  ) else (
    echo (failed list file missing) >> "%LOG_FILE%"
  )
)

echo Done. Log written to: "%LOG_FILE%"
goto :END_OK

:END_OK
echo.
echo SUCCESS. Log written to: "%LOG_FILE%"
echo.
if exist "%LOCK_FILE%" del /q "%LOCK_FILE%" >nul 2>&1
pause
exit /b 0

:END_FAIL
echo.
echo FAILED. See log: "%LOG_FILE%"
echo.
if exist "%LOCK_FILE%" del /q "%LOCK_FILE%" >nul 2>&1
pause
exit /b 1
