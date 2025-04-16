:: This script runs the relevant cmake commands for building the Piquasso project on Windows.

@echo off
setlocal

:: Step 1: Delete the build directory if it exists
if exist build (
    echo Deleting existing build directory...
    rmdir /s /q build
)

:: Step 2: Run CMake configuration
echo Running CMake configuration...
cmake -B build -DCMAKE_INSTALL_PREFIX="%cd%" -DPYBIND11_FINDPYTHON=ON

:: Step 3: Build the project
echo Building project...
cmake --build build --config Debug

:: Step 4: Install the project
echo Installing project...
cmake --install build --config Debug

echo Done.
endlocal