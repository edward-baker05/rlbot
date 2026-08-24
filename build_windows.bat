@echo off
setlocal enabledelayedexpansion

echo =======================================================
echo Building DashBot for Windows (MSVC)
echo =======================================================

set "REPO=%~dp0"
set "BUILD_DIR=%REPO%bot\build"

:: 1. Ensure LibTorch is available
if not exist "%REPO%libs\libtorch\include" (
    echo [INFO] Downloading Windows LibTorch (Release CPU)...
    powershell -Command "Invoke-WebRequest -Uri 'https://download-r2.pytorch.org/libtorch/cpu/libtorch-win-shared-with-deps-2.2.0%%2Bcpu.zip' -OutFile '%REPO%libs\libtorch-win.zip'"
    powershell -Command "Expand-Archive -Path '%REPO%libs\libtorch-win.zip' -DestinationPath '%REPO%libs'"
    del "%REPO%libs\libtorch-win.zip"
)

:: 2. Configure with CMake
echo [INFO] Configuring CMake...
cmake -B "%BUILD_DIR%" -S "%REPO%bot" -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="%REPO%libs\libtorch"

:: 3. Build DashBot
echo [INFO] Compiling DashBot...
cmake --build "%BUILD_DIR%" --config Release --target DashBot

:: 4. Copy to dist\Dash
if exist "%BUILD_DIR%\Release\DashBot.exe" (
    copy /y "%BUILD_DIR%\Release\DashBot.exe" "%REPO%dist\Dash\"
    echo [SUCCESS] Copied DashBot.exe to dist\Dash\
) else if exist "%BUILD_DIR%\DashBot.exe" (
    copy /y "%BUILD_DIR%\DashBot.exe" "%REPO%dist\Dash\"
    echo [SUCCESS] Copied DashBot.exe to dist\Dash\
) else (
    echo [ERROR] Build did not produce DashBot.exe. Please check compiler output above.
    pause
    exit /b 1
)

echo =======================================================
echo Build complete! Your dist\Dash folder is ready to zip and send.
echo =======================================================
pause
