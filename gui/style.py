"""Light theme for the whole application."""

QSS = """
QWidget { font-size: 10pt; color: #1F2328; }
QMainWindow, QWidget#home { background: #F6F8FA; }
QTabWidget::pane { border: none; }
QTabBar::tab { padding: 9px 20px; margin-right: 2px; background: transparent; color: #57606A;
               border-bottom: 3px solid transparent; font-weight: 600; }
QTabBar::tab:selected { color: #1F4E79; border-bottom: 3px solid #1F4E79; }
QTabBar::tab:hover { color: #1F2328; }
QPushButton, QToolButton { padding: 6px 14px; border: 1px solid #D0D7DE; border-radius: 6px; background: #FFFFFF; }
QPushButton:hover, QToolButton:hover { background: #F3F4F6; }
QPushButton:disabled, QToolButton:disabled { color: #8C959F; background: #F6F8FA; }
QPushButton#primary { background: #1F4E79; color: white; border-color: #1F4E79; font-weight: 600; }
QPushButton#primary:hover { background: #2A6299; }
QPushButton#primary:disabled { background: #9DB3C8; border-color: #9DB3C8; }
QPushButton#segLeft, QPushButton#segRight { padding: 5px 16px; }
QPushButton#segLeft { border-top-right-radius: 0; border-bottom-right-radius: 0; }
QPushButton#segRight { border-top-left-radius: 0; border-bottom-left-radius: 0; border-left: none; }
QPushButton#segLeft:checked, QPushButton#segRight:checked { background: #1F4E79; color: white; border-color: #1F4E79; }
QGroupBox { border: 1px solid #D0D7DE; border-radius: 8px; margin-top: 10px; background: #FFFFFF; }
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; color: #57606A; }
QListWidget { border: 1px solid #D0D7DE; border-radius: 8px; background: #FFFFFF; padding: 4px; }
QListWidget::item { padding: 6px 4px; border-bottom: 1px solid #EEF1F4; }
QListWidget::item:selected { background: #DDEBF7; color: #1F2328; }
QTableWidget { border: 1px solid #D0D7DE; border-radius: 6px; background: #FFFFFF;
               alternate-background-color: #F6F8FA; gridline-color: transparent; }
QHeaderView::section { background: #FFFFFF; border: none; border-bottom: 2px solid #1F4E79; padding: 5px;
                       font-weight: 600; color: #1F4E79; }
QFrame#card { background: #FFFFFF; border: 1px solid #D0D7DE; border-radius: 10px; padding: 6px; }
QLabel#cardTitle { font-size: 12pt; font-weight: 700; }
QLabel#appName { font-size: 26pt; font-weight: 800; color: #1F4E79; }
QLabel#appSub { font-size: 12.5pt; color: #33383F; }
QLabel#intro { font-size: 10.5pt; line-height: 140%; }
QLabel#section { font-size: 11pt; font-weight: 700; color: #1F4E79; padding: 4px 0; }
QLabel#muted, QLabel#legendText { color: #656D76; font-size: 9pt; }
QLabel#bigNumber { font-size: 30pt; font-weight: 800; color: #1F2328; }
QProgressBar { border: 1px solid #D0D7DE; border-radius: 6px; background: #FFFFFF; text-align: center; height: 20px; }
QProgressBar::chunk { background: #4C8BC2; border-radius: 5px; }
QScrollArea { background: transparent; }
"""
