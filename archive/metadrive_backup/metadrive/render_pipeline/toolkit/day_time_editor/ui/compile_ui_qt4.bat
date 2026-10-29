@echo off

rem Adjust the path to match your PyQt4 installation
SET PYQTPATH="C:\Projekte\Panda3D\built_x64\python\Lib\site-packages\PyQt4"

python -m PyQt4.uic.pyuic --from-imports main_window.ui -o main_window_generated.py
python -m PyQt4.uic.pyuic --from-imports point_insert.ui -o point_insert_dialog_generated.py
"%PYQTPATH%\pyrcc4.exe" -py3 resources.qrc -o resources_rc.py

pause
