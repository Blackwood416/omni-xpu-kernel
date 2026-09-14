@echo off
call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat" --force >nul
cd /d "%~dp0..\benchmarks"
icpx -fsycl -fsycl-targets=spir64_gen -Xs "-device dg2" -O2 dg2_convrot_standalone.cpp -o dg2_convrot_standalone.exe
exit /b %errorlevel%
