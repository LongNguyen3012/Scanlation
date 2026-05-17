@echo off
echo ============================================================
echo  Comic Translator - Build .exe with PyInstaller
echo ============================================================
echo.
echo Installing PyInstaller if not already installed...
pip install pyinstaller

echo.
echo Building executable...
echo (This will take a few minutes and produce a large file ~400-600 MB)
echo.

pyinstaller ^
  --onefile ^
  --windowed ^
  --name "ComicTranslator" ^
  --icon NONE ^
  --hidden-import "easyocr" ^
  --hidden-import "deep_translator" ^
  --hidden-import "cv2" ^
  --hidden-import "PIL" ^
  --collect-all "easyocr" ^
  --collect-all "deep_translator" ^
  main.py

echo.
echo ============================================================
echo  Done!  Your .exe is in the  dist\  folder.
echo.
echo  IMPORTANT: The first time the .exe runs it will download
echo  EasyOCR model files (~200 MB) to  %%USERPROFILE%%\.EasyOCR\
echo  This only happens once.
echo ============================================================
pause
