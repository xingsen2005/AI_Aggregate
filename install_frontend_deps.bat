@echo off

REM Install frontend dependencies and optionally start development server

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

REM Ask user if they want to start the development server
echo.
echo Do you want to start the development server? (Y/N)
set /p choice=

if /i "%choice%"=="Y" (
    echo Starting development server...
    echo This will run 'npm run dev' which starts the Vite development server.
    echo The server will be available at http://localhost:5173 (default port)
    echo Press Ctrl+C to stop the server.
    echo.
    npm run dev
    
    if %errorlevel% neq 0 (
        echo Error: Failed to start the development server.
        pause
        exit /b 1
    )
) else (
    echo Development server not started.
    echo You can start it manually later by running 'npm run dev' in the frontend directory.
    pause
)

cd ..