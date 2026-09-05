@echo off
title Push Project to GitHub - Prajakta Pawar
color 0A

echo ========================================================
echo  UPLOADING PRODUCT PRICE FORECASTING TO GITHUB
echo  GitHub Username: prajaktasatyawanpawar30-sys
echo ========================================================
echo.

where git >nul 2>nul
if %ERRORLEVEL% neq 0 (
    color 0C
    echo [ERROR] Git is not installed or not found in PATH!
    echo Please install Git from: https://git-scm.com/downloads/win
    echo Then run this file again.
    echo.
    pause
    exit /b
)

echo [1/5] Initializing Git repository...
git init

echo.
echo [2/5] Staging all files and directories...
git add .

echo.
echo [3/5] Creating commit...
git commit -m "College Project: Product Price Forecasting with Explainability"

echo.
echo [4/5] Setting branch to main...
git branch -M main

echo.
echo [5/5] Connecting to GitHub repository...
git remote remove origin >nul 2>nul
git remote add origin https://github.com/prajaktasatyawanpawar30-sys/Product-Price-Forecasting-with-Explainability.git

echo.
echo ========================================================
echo Pushing files to GitHub...
echo (A browser popup may appear asking you to Sign In to GitHub.
echo  Click 'Sign in with your browser' and authorize it.)
echo ========================================================
echo.

git push -u origin main

echo.
if %ERRORLEVEL% equ 0 (
    color 0A
    echo ========================================================
    echo  SUCCESS! Your project has been uploaded to GitHub!
    echo  View it at:
    echo  https://github.com/prajaktasatyawanpawar30-sys/Product-Price-Forecasting-with-Explainability
    echo ========================================================
) else (
    color 0E
    echo.
    echo If push failed, your GitHub repository might already have a file,
    echo or authentication is required. Retrying with force push...
    echo.
    git push -u origin main --force
)

echo.
pause
