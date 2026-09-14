@echo off
REM Local A770/DG2 wheel build helper for PyTorch 2.14 (uses isolated uv venv).
call "C:\Program Files\Microsoft Visual Studio\18\Community\Common7\Tools\VsDevCmd.bat" -arch=amd64 -host_arch=amd64
if errorlevel 1 exit /b 1
call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat" --force
if errorlevel 1 exit /b 1

set "DNNLROOT=C:\Program Files (x86)\Intel\oneAPI\dnnl\latest"
set "OMNI_XPU_DEVICE=a770"
set "OMNI_XPU_REQUIRE_CUTE=0"
set "OMNI_XPU_BUILD_JOBS=24"
set "PATH=C:\Users\Administrator\AppData\Local\Temp\omni-build-torch214\Lib\site-packages\torch\lib;%PATH%"

cd /d "%~dp0.."
"C:\Users\Administrator\AppData\Local\Temp\omni-build-torch214\Scripts\python.exe" -m pip wheel . --no-build-isolation --no-deps --wheel-dir dist
exit /b %errorlevel%
