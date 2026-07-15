import sys

from PySide6.QtWidgets import QApplication, QMessageBox
from ui_components.main_window import MainWindow
from module_manager import ModuleManager

# Global instance for easier access (e.g., from module's control widgets)
module_manager_instance = ModuleManager()

def run_app():
    app = QApplication(sys.argv)
    
    # Initialize the global data store (optional, can be done on first use)
    #_ = ImageDataStore()

    main_window = MainWindow(module_manager_instance)
    main_window.show()

    sys.exit(app.exec())

if __name__ == "__main__":
    run_app()
