@echo off

python -m PyQt5.uic.pyuic --from-imports main_window.ui -o main_window_generated.py
python -m PyQt5.uic.pyuic --from-imports point_insert.ui -o point_insert_dialog_generated.py
python -m PyQt5.pyrcc_main resources.qrc -o resources_rc.py

pause
