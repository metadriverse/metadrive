@echo off

"C:/mitsuba/mitsuba" -p 6 res/scene.xml
echo ""
echo Rendering done
echo ""
echo ""
echo ""
"C:/mitsuba/mtsutil" tonemap res/scene.exr
copy res\scene.png scene.png
del res\scene.exr
del mitsuba.*.log
