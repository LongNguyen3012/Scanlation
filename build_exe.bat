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
  --onedir ^
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
echo  Done!  Your app folder is at:  dist\ComicTranslator\
echo  Share the entire ComicTranslator folder — not just the .exe.
echo.
echo  Settings and downloaded models are saved to:
echo  %%APPDATA%%\ComicTranslator\
echo.
echo  IMPORTANT: The first time the app runs it will download
echo  EasyOCR model files (~200 MB) to  %%USERPROFILE%%\.EasyOCR\
echo  This only happens once.
echo ============================================================
pause
