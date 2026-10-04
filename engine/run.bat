@echo off
rem Play the show: preview window, Spout sender "MuonBloom". Arguments are passed on, for example
rem     engine\run.bat --paused --detectors live      the show: waits for the time sent by OSC, live detectors
rem     engine\run.bat --paused --detectors live --output 2     ... and the raster on display 2 (the HDMI output);
rem                                                   without --output: the display chosen in the OUTPUT panel last time
rem     engine\run.bat --from 180 --paused
rem     engine\run.bat --loop --detectors live
rem See the top of engine\src\main.cpp (or engine\BRIEF.md, section 11) for the options and the keys.
rem
rem If the engine stops by itself while it runs (a crash, the graphics driver reset: exit code 3 or a crash
rem code), it is started again where it was. It is not started again when somebody closed it (0) or when it
rem cannot start as it is set up (2: missing sound files, Python not found ...).
rem Messages are also written to engine\out\engine.log.
setlocal
set "HERE=%~dp0"
if not exist "%HERE%build\muonengine.exe" call "%HERE%build.bat" || exit /b 1
if not exist "%HERE%out" mkdir "%HERE%out"
set "POS=%HERE%out\position.txt"
if exist "%POS%" del "%POS%"
set "RESUME="

:again
"%HERE%build\muonengine.exe" live %RESUME% %* --position-file "%POS%"
set "RC=%ERRORLEVEL%"
if "%RC%"=="0" exit /b 0
if "%RC%"=="2" exit /b 2
echo.
echo muonengine stopped with code %RC%: starting it again in 3 seconds (Ctrl+C to give up)
echo %DATE% %TIME%  muonengine stopped with code %RC%: restarted by run.bat>> "%HERE%out\engine.log"
timeout /t 3 /nobreak >nul
set "RESUME="
if exist "%POS%" (
    set /p T=<"%POS%"
    call set "RESUME=--from %%T%%"
)
goto again
