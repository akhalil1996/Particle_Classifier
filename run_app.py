"""Start the CauliQuant desktop application."""
import multiprocessing
import os
import sys
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


def self_test():
    """Run without a window; exit code 0 if the analysis machinery works (used by the build)."""
    import tempfile
    import traceback
    log_path = os.path.join(tempfile.gettempdir(), "cauliquant_selftest.log")
    with open(log_path, "w") as log:
        try:
            from cauliquant.selftest import run
            run(log=lambda m: (log.write(m + "\n"), log.flush()))
            log.write("SELF-TEST PASSED\n")
            return 0
        except Exception:
            log.write(traceback.format_exc())
            return 1


def main():
    if "--self-test" in sys.argv:
        sys.exit(self_test())
    from PySide6.QtWidgets import QApplication
    from PySide6.QtGui import QIcon
    from cauliquant import APP_NAME
    from cauliquant.resources import resource
    from gui.style import QSS
    from gui.main_window import MainWindow
    app = QApplication(sys.argv)
    app.setOrganizationName("McGill")
    app.setApplicationName(APP_NAME)
    app.setStyle("Fusion")
    app.setStyleSheet(QSS)
    ico = resource("assets", "icon.png")
    if os.path.exists(ico):
        app.setWindowIcon(QIcon(ico))
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
