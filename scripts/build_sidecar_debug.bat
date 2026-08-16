@echo off
call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat" --force >nul
cd /d "%~dp0..\omni_xpu_kernel\lgrf_uni"
if not exist build mkdir build
icpx -fsycl -fsycl-targets=spir64_gen -Xs "-device dg2 -options -doubleGRF" -O2 -DNDEBUG -std=c++17 -DNOMINMAX -DWIN32_LEAN_AND_MEAN -DBUILD_ESIMD_KERNEL_LIB -DOMNI_XPU_ARCH_DG2=1 -shared -o build\lgrf_sdp_debug.pyd sdp_kernels.cpp
exit /b %errorlevel%
