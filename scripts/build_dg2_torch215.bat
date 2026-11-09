@echo off
REM Local A770/DG2 wheel build helper for PyTorch 2.15 (uses isolated uv venv).
call "C:\Program Files\Microsoft Visual Studio\18\Community\Common7\Tools\VsDevCmd.bat" -arch=amd64 -host_arch=amd64
if errorlevel 1 exit /b 1

REM Initialize Intel oneAPI 2026.2 environment
call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat" --force

where icx-cl >nul 2>&1
if errorlevel 1 (
    set "ONEAPI_ROOT=C:\Program Files (x86)\Intel\oneAPI"
    set "ONEAPI_COMPILER_ROOT=C:\Program Files (x86)\Intel\oneAPI\compiler\latest"
    set "PATH=C:\Program Files (x86)\Intel\oneAPI\compiler\latest\bin;C:\Program Files (x86)\Intel\oneAPI\compiler\latest\bin\compiler;C:\Program Files (x86)\Intel\oneAPI\tbb\latest\bin;%PATH%"
    set "INCLUDE=C:\Program Files (x86)\Intel\oneAPI\compiler\latest\include;C:\Program Files (x86)\Intel\oneAPI\compiler\latest\include\sycl;%INCLUDE%"
    set "LIB=C:\Program Files (x86)\Intel\oneAPI\compiler\latest\lib;C:\Program Files (x86)\Intel\oneAPI\compiler\latest\opt\compiler\lib;C:\Program Files (x86)\Intel\oneAPI\tbb\latest\lib;%LIB%"
)

where icx-cl >nul 2>&1
if errorlevel 1 (
    echo icx-cl.exe is not on PATH after configuring oneAPI. 1>&2
    exit /b 1
)

set "DNNLROOT=C:\Program Files (x86)\Intel\oneAPI\dnnl\latest"
set "OMNI_XPU_DEVICE=a770"
set "OMNI_XPU_REQUIRE_CUTE=0"
set "OMNI_XPU_BUILD_JOBS=24"
set "OMNI_XPU_BUILD_NUMBER=1"
set "PATH=C:\Users\Administrator\AppData\Local\Temp\omni-build-torch215\Lib\site-packages\torch\lib;%PATH%"

cd /d "%~dp0.."
"C:\Users\Administrator\AppData\Local\Temp\omni-build-torch215\Scripts\python.exe" -m pip wheel . --no-build-isolation --no-deps --wheel-dir dist
exit /b %errorlevel%
