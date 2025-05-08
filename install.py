import sys
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QPushButton,
    QVBoxLayout, QHBoxLayout, QLabel, QStackedWidget, QFrame,
    QCheckBox, QProgressBar
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QIcon
import time
import os


class ModernTile(QFrame):
    def __init__(self, icon: str, title: str, callback=None):
        super().__init__()
        self.completed = False
        self.arrow = QLabel("✅")
        self.arrow.setStyleSheet("font-size: 40px; color: green;")
        self.arrow.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.arrow.setVisible(False)

        self.setStyleSheet("""
            QFrame {
                background-color: #f9f9f9;
                border-radius: 16px;
                padding: 24px;
                border: 1px solid #ddd;
            }
        """)
        layout = QVBoxLayout(self)

        icon_label = QLabel(icon)
        icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        icon_label.setStyleSheet("font-size: 36px;")

        title_label = QLabel(title)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet("font-size: 18px; font-weight: 600;")

        layout.addWidget(icon_label)
        layout.addWidget(title_label)
        layout.addWidget(self.arrow)

        if callback:
            self.button = QPushButton("Start")
            self.button.setFixedHeight(36)
            self.button.setStyleSheet("""
                QPushButton {
                    background-color: #4CAF50;
                    color: white;
                    border: none;
                    border-radius: 18px;
                    font-size: 14px;
                    padding: 8px 24px;
                }
                QPushButton:hover {
                    background-color: #45a049;
                }
            """)
            self.button.clicked.connect(callback)
            layout.addWidget(self.button)

    def mark_complete(self):
        self.completed = True
        self.setStyleSheet("""
            QFrame {
                background-color: #c8f7c5;
                border-radius: 16px;
                padding: 24px;
                border: 1px solid #5cb85c;
            }
        """)
        self.arrow.setVisible(True)


class ModernMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PAM Authenticator Setup")
        self.setGeometry(100, 100, 850, 600)

        self.skip_camera = False
        self.skip_fingerprint = False
        self.installation_done = False

        self.fingerprint_done = False
        self.face_day_done = False
        self.face_night_done = False

        container = QWidget()
        self.setCentralWidget(container)
        main_layout = QVBoxLayout(container)

        self.stack = QStackedWidget()
        main_layout.addWidget(self.stack)

        nav_layout = QHBoxLayout()
        self.prev_btn = QPushButton("◀")
        self.next_btn = QPushButton("▶")
        for btn in (self.prev_btn, self.next_btn):
            btn.setFixedSize(50, 50)
            btn.setStyleSheet("""
                QPushButton {
                    font-size: 20px;
                    background-color: #e0e0e0;
                    border-radius: 25px;
                }
                QPushButton:hover {
                    background-color: #d5d5d5;
                }
            """)
        self.prev_btn.clicked.connect(self.prev_page)
        self.next_btn.clicked.connect(self.next_page)

        nav_layout.addStretch()
        nav_layout.addWidget(self.prev_btn)
        nav_layout.addWidget(self.next_btn)
        nav_layout.addStretch()

        main_layout.addLayout(nav_layout)

        # Pages
        self.stack.addWidget(self.create_install_page())
        self.fingerprint_page = self.create_fingerprint_page()
        self.stack.addWidget(self.fingerprint_page)
        self.face_page = self.create_face_page()
        self.stack.addWidget(self.face_page)

        self.update_nav_buttons()

    def create_install_page(self):
        page = QWidget()
        page.setStyleSheet("background-color: white;")
        layout = QVBoxLayout(page)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        tile_layout = QHBoxLayout()
        self.fingerprint_tile = ModernTile("🫆", "Fingerprint sensor dependencies")
        self.face_tile = ModernTile("😶", "Face recognition dependencies")
        tile_layout.addWidget(self.fingerprint_tile)
        tile_layout.addWidget(self.face_tile)
        layout.addLayout(tile_layout)

        self.no_camera_cb = QCheckBox("I don't have a camera")
        self.no_fingerprint_cb = QCheckBox("I don't have a fingerprint sensor")
        layout.addWidget(self.no_camera_cb)
        layout.addWidget(self.no_fingerprint_cb)

        install_btn = QPushButton("Start Installation")
        install_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-size: 16px;
                padding: 10px 20px;
                border: none;
                border-radius: 20px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        install_btn.clicked.connect(self.on_install)
        layout.addWidget(install_btn, alignment=Qt.AlignmentFlag.AlignCenter)

        return page

    def create_fingerprint_page(self):
        page = QWidget()
        page.setStyleSheet("background-color: white;")
        layout = QVBoxLayout(page)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setSpacing(20)

        # Title
        title = QLabel("Fingerprint Registration")
        title.setStyleSheet("""
            QLabel {
                font-size: 24px;
                font-weight: bold;
                color: #333;
                margin-bottom: 20px;
            }
        """)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        # Fingerprint icon with animation placeholder
        self.fingerprint_icon = QLabel("🫴")
        self.fingerprint_icon.setStyleSheet("""
            QLabel {
                font-size: 100px;
                margin: 20px 0;
            }
        """)
        self.fingerprint_icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.fingerprint_icon)

        # Instructions
        instructions = QLabel("Place your finger on the sensor\nand lift it repeatedly until complete")
        instructions.setStyleSheet("""
            QLabel {
                font-size: 16px;
                color: #555;
                text-align: center;
                margin-bottom: 20px;
            }
        """)
        instructions.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(instructions)

        # Start button
        start_btn = QPushButton("Start Registration")
        start_btn.setFixedSize(200, 50)
        start_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-size: 16px;
                border: none;
                border-radius: 25px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """)
        start_btn.clicked.connect(self.on_fingerprint)
        layout.addWidget(start_btn, alignment=Qt.AlignmentFlag.AlignCenter)

        # Progress bar
        progress = QProgressBar()
        progress.setFixedWidth(400)
        progress.setRange(0, 100)
        progress.setValue(0)
        progress.setStyleSheet("""
            QProgressBar {
                border: 2px solid #d3d3d3;
                border-radius: 10px;
                text-align: center;
                height: 20px;
                margin-top: 20px;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
                border-radius: 8px;
            }
        """)
        layout.addWidget(progress, alignment=Qt.AlignmentFlag.AlignCenter)

        # Status text
        self.fingerprint_status = QLabel("Ready to start fingerprint registration")
        self.fingerprint_status.setStyleSheet("""
            QLabel {
                font-size: 14px;
                color: #666;
                margin-top: 10px;
            }
        """)
        self.fingerprint_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.fingerprint_status)

        page.progress = progress
        page.start_btn = start_btn
        return page

    def create_face_page(self):
        page = QWidget()
        page.setStyleSheet("background-color: white;")
        layout = QVBoxLayout(page)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setSpacing(20)

        # Title
        title = QLabel("Face Recognition Training")
        title.setStyleSheet("""
            QLabel {
                font-size: 24px;
                font-weight: bold;
                color: #333;
                margin-bottom: 20px;
            }
        """)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        # Buttons container
        buttons_container = QWidget()
        buttons_layout = QVBoxLayout(buttons_container)
        buttons_layout.setSpacing(15)
        buttons_layout.setContentsMargins(0, 0, 0, 0)

        # Day button
        day_btn = QPushButton()
        day_btn.setFixedSize(350, 120)
        day_btn.setStyleSheet("""
            QPushButton {
                background-color: #f0f8ff;
                border-radius: 12px;
                border: 2px solid #add8e6;
                font-size: 16px;
                padding: 15px;
                text-align: center;
            }
            QPushButton:hover {
                background-color: #e1f0fa;
            }
        """)
        day_layout = QVBoxLayout(day_btn)
        day_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        day_icon = QLabel("☀️")
        day_icon.setStyleSheet("font-size: 36px;")
        day_icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        day_text = QLabel("Normal Lighting Conditions")
        day_text.setStyleSheet("font-size: 16px; font-weight: 600;")
        day_text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        day_layout.addWidget(day_icon)
        day_layout.addWidget(day_text)
        day_btn.clicked.connect(self.on_face_day)

        # Night button
        night_btn = QPushButton()
        night_btn.setFixedSize(350, 120)
        night_btn.setStyleSheet("""
            QPushButton {
                background-color: #f0f0ff;
                border-radius: 12px;
                border: 2px solid #b0c4de;
                font-size: 16px;
                padding: 15px;
                text-align: center;
            }
            QPushButton:hover {
                background-color: #e6e6fa;
            }
        """)
        night_layout = QVBoxLayout(night_btn)
        night_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        night_icon = QLabel("🌙")
        night_icon.setStyleSheet("font-size: 36px;")
        night_icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        night_text = QLabel("Low Lighting Conditions")
        night_text.setStyleSheet("font-size: 16px; font-weight: 600;")
        night_text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        night_layout.addWidget(night_icon)
        night_layout.addWidget(night_text)
        night_btn.clicked.connect(self.on_face_night)

        buttons_layout.addWidget(day_btn, alignment=Qt.AlignmentFlag.AlignCenter)
        buttons_layout.addWidget(night_btn, alignment=Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(buttons_container)

        # Progress bar
        progress = QProgressBar()
        progress.setFixedWidth(400)
        progress.setRange(0, 100)
        progress.setValue(0)
        progress.setStyleSheet("""
            QProgressBar {
                border: 2px solid #d3d3d3;
                border-radius: 10px;
                text-align: center;
                height: 20px;
                margin-top: 20px;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
                border-radius: 8px;
            }
        """)
        layout.addWidget(progress, alignment=Qt.AlignmentFlag.AlignCenter)

        # Status text
        self.status_label = QLabel("Complete both trainings for best results")
        self.status_label.setStyleSheet("""
            QLabel {
                font-size: 14px;
                color: #666;
                margin-top: 10px;
            }
        """)
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.status_label)

        page.progress = progress
        return page

    def update_nav_buttons(self):
        idx = self.stack.currentIndex()
        self.prev_btn.setEnabled(idx > 0)

        if idx == 1:
            self.next_btn.setEnabled(self.fingerprint_done)
        elif idx == 2:
            self.next_btn.setEnabled(self.face_day_done and self.face_night_done)
        else:
            self.next_btn.setEnabled(self.installation_done)

    def next_page(self):
        idx = self.stack.currentIndex()
        if idx < self.stack.count() - 1:
            self.stack.setCurrentIndex(idx + 1)
        self.update_nav_buttons()

    def prev_page(self):
        idx = self.stack.currentIndex()
        if idx > 0:
            self.stack.setCurrentIndex(idx - 1)
        self.update_nav_buttons()

    def on_install(self):
        if not self.no_fingerprint_cb.isChecked():
            if not self.no_camera_cb.isChecked(): # install the final version + fprintd
                
                os.system("yay -S fprintd")# Arch-linux
                
                self.fingerprint_tile.mark_complete()
                self.face_tile.mark_complete()
                pass
            else : # just install pam-fprint-grosshack + fprintd
                os.system("yay -S fprintd")# Arch-linux
                os.system("yay -S pam-fprint-grosshack")# Arch-linux
                self.fingerprint_tile.mark_complete()
                pass
        elif not self.no_camera_cb.isChecked(): # install the pam-camera-grosshack version
            
            self.face_tile.mark_complete()
            pass
        if self.no_fingerprint_cb.isChecked() or self.fingerprint_tile.completed:
            if self.no_camera_cb.isChecked() or self.face_tile.completed:
                self.installation_done = True

        self.update_nav_buttons()

        

    def on_fingerprint(self):
        print("Starting fingerprint registration...")
        self.fingerprint_page.start_btn.setEnabled(False)
        self.fingerprint_status.setText("Scanning your fingerprint...")
        self.fingerprint_status.setStyleSheet("color: #4CAF50; font-weight: bold;")
        self.progress_value = 0
        self.update_fingerprint_progress()
    def update_fingerprint_progress(self):
        # appelle la fonction qui enregistre le doigt 
        # il faut réussir à mettre une bonne représentation de la progression de l'enregistrement (en fonction des logs de fprintd) 
        self.fingerprint_icon.setText("🫳")
        while self.progress_value < 100:
            self.progress_value += 2   # a changer en fonction de la progression de fprintd
            self.fingerprint_page.progress.setValue(self.progress_value)
            time.sleep(0.1)

            if self.progress_value >= 100:
                self.fingerprint_done = True
                self.fingerprint_status.setText("Fingerprint registration complete!")
                self.fingerprint_icon.setText("👍")
                self.update_nav_buttons()
                break

    def on_face_day(self):
        print("First face recognition training done.")
        # appelle la fonction qui fait le training
        self.face_day_done = True
        self.update_face_progress()

    def on_face_night(self):
        print("Last face recognition training done.")
        # appelle la fonction qui fait le training
        self.face_night_done = True
        self.update_face_progress()

    def update_face_progress(self):
        progress = 0
        if self.face_day_done:
            progress += 50
        if self.face_night_done:
            progress += 50
        self.face_page.progress.setValue(progress)
        
        if progress == 100:
            self.status_label.setText("Training completed successfully!")
            self.status_label.setStyleSheet("color: #4CAF50; font-size: 14px; font-weight: bold;")
        elif progress == 50:
            self.status_label.setText("One training remaining")
        else:
            self.status_label.setText("Complete both trainings for best results")
            
        self.update_nav_buttons()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ModernMainWindow()
    window.show()
    sys.exit(app.exec())