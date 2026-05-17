import sys
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt
from app.main_window import MainWindow


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Comic Translator")
    app.setStyle("Fusion")

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
