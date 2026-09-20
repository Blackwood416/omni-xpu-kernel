@echo off
REM Local A770/DG2 wheel build helper for PyTorch 2.14 (uses isolated uv venv).
call "C:\Program Files\Microsoft Visual Studio\18\Community\Common7\Tools\VsDevCmd.bat" -arch=amd64 -host_arch=amd64
if errorlevel 1 exit /b 1
call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat" --force
REM setvars.bat can fail to run its component vars.bat files on this machine
REM ("'vars.bat' is not recognized"), which leaves icx off PATH. Fall back to
REM the explicit compiler layout instead of trusting its exit code.
where icx-cl >nul 2>&1
if errorlevel 1 (
    set "ONEAPI_COMPILER_ROOT=C:\Program Files (x86)\Intel\oneAPI\compiler\latest"
    set "PATH=C:\Program Files (x86)\Intel\oneAPI\compiler\latest\bin;%PATH%"
    set "INCLUDE=C:\Program Files (x86)\Intel\oneAPI\compiler\latest\include;C:\Program Files (x86)\Intel\oneAPI\compiler\latest\include\sycl;%INCLUDE%"
    set "LIB=C:\Program Files (x86)\Intel\oneAPI\compiler\latest\lib;C:\Program Files (x86)\Intel\oneAPI\compiler\latest\opt\compiler\lib;%LIB%"
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
set "PATH=C:\Users\Administrator\AppData\Local\Temp\omni-build-torch214\Lib\site-packages\torch\lib;%PATH%"

cd /d "%~dp0.."
"C:\Users\Administrator\AppData\Local\Temp\omni-build-torch214\Scripts\python.exe" -m pip wheel . --no-build-isolation --no-deps --wheel-dir dist
exit /b %errorlevel%
