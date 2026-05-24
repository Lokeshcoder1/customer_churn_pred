@echo off
REM This script fixes the .gitignore issue

echo Fixing .gitignore...

REM Replace the corrupted .gitignore with the clean one
del .gitignore
ren .gitignore_new .gitignore

echo .gitignore has been fixed.

REM Now remove cached files that should be ignored
echo.
echo Removing files from git cache that should be ignored...

REM Remove data directory
git rm -r --cached data/ 2>nul

REM Remove models directory
git rm -r --cached models/ 2>nul

REM Remove artifacts directory
git rm -r --cached artifacts/ 2>nul

REM Remove venv directory
git rm -r --cached venv/ 2>nul

REM Remove logs directory
git rm -r --cached logs/ 2>nul

REM Remove __pycache__ directories
git rm -r --cached __pycache__/ 2>nul

REM Remove .pytest_cache
git rm -r --cached .pytest_cache/ 2>nul

REM Refresh git index
echo.
echo Refreshing git index...
git add .

echo.
echo Done! Now commit these changes with:
echo git commit -m "Fix: Remove ignored files from git tracking"
