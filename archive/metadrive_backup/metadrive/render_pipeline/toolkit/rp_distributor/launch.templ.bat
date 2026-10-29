@echo off

rem set required paths
set PANDAPATH=%~dp0panda3d
set PATH=%PANDAPATH%\bin;%PANDAPATH%;%PATH%
set PYTHONPATH=%PANDAPATH%\;%PYTHONPATH%

rem check for visual studio dlls
where msvcp140.dll >nul 2>nul
if ERRORLEVEL 1 (
   @echo This program requires the Microsoft Visual C++ 2015 Runtime.
   @echo Hit a key to visit the download page.
   pause
   start https://www.microsoft.com/en-us/download/details.aspx?id=48145
   goto end
)

rem execute actual program
@echo Running Panda3D from "%PANDAPATH%"
"%PANDAPATH%\python\ppython.exe" -B application/main.py
pause

:end
