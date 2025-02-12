import sys
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow
from PyQt5.QtCore import Qt

# Subclass QMainWindow to customize your main window
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Minha Primeira Aplicação PyQt5")

        label = QLabel("Olá, PyQt5!")
        label.setAlignment(Qt.AlignCenter)

        self.setCentralWidget(label)

# Inicializa a aplicação
app = QApplication(sys.argv)

window = MainWindow()
window.show()

# Executa o loop de eventos da aplicação
sys.exit(app.exec_())
