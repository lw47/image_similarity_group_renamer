@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
echo ========================================
echo Image Similarity Grouping Tool
echo Script: %~dp0image_similarity_renamer.py
echo ========================================
where python >nul 2>&1
if errorlevel 1 (
    echo Python was not found in PATH.
    pause
    exit /b 1
)
python "%~dp0image_similarity_renamer.py"
set ERR=%ERRORLEVEL%
echo.
echo Process exited with code %ERR%.
pause
endlocal
