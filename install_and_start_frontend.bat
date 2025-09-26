@echo off

REM This script installs frontend dependencies and starts the development server automatically

echo Installing frontend dependencies...
cd frontend

REM Check if npm is installed
where npm >nul 2>nul
if %errorlevel% neq 0 (
    echo Error: npm not found. Please install Node.js first.
    echo Node.js download URL: https://nodejs.org/
    pause
    exit /b 1
)

REM Install dependencies
npm install

if %errorlevel% neq 0 (
    echo Error: Failed to install dependencies.
    pause
    exit /b 1
)

echo Frontend dependencies installed successfully!

REM Start the development server
echo Starting development server...
echo The server will be available at http://localhost:3000
echo Press Ctrl+C to stop the server.
echo.
npm run dev


if %errorlevel% neq 0 (
    echo Error: Failed to start the development server.
    pause
    exit /b 1
)

cd ..