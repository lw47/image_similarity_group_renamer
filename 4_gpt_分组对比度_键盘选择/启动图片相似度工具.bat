@echo off
chcp 65001 >nul
cd /d "%~dp0"
python "%~dp0image_similarity_renamer.py"
set ERR=%ERRORLEVEL%
echo.
echo Process exited with code %ERR%.
if not "%ERR%"=="0" pause
