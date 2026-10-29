@echo off

"C:/mitsuba/mitsuba" -p 6 res/scene.xml
echo ""
echo Rendering done
echo ""
echo ""
echo ""
"C:/mitsuba/mtsutil" tonemap -g 2.0 res/scene.exr
copy res\scene.png scene.png
