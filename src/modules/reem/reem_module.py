from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QSlider, QPushButton, QComboBox, QStackedWidget, QDoubleSpinBox, QGridLayout
from PySide6.QtCore import Qt, Signal
import numpy as np
import imageio # For general image loading (can use Pillow too)
import skimage.filters
import skimage.morphology
from skimage.color import rgb2gray
from scipy.ndimage import convolve

from modules.i_image_module import IImageModule
from image_data_store import ImageDataStore

# --- Parameter Widgets for Different Operations ---
class BaseParamsWidget(QWidget):
    """Base class for parameter widgets to ensure a consistent interface."""
    def get_params(self) -> dict:
        raise NotImplementedError

class NoParamsWidget(BaseParamsWidget):
    """A placeholder widget for operations with no parameters."""
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        label = QLabel("This operation has no parameters.")
        label.setStyleSheet("font-style: italic; color: gray;")
        layout.addWidget(label)
        layout.addStretch()

    def get_params(self) -> dict:
        return {}

class GaussianParamsWidget(BaseParamsWidget):
    """A widget for Gaussian blur parameters."""
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QLabel("Sigma (Standard Deviation):"))
        self.sigma_spinbox = QDoubleSpinBox()
        self.sigma_spinbox.setMinimum(0.1)
        self.sigma_spinbox.setMaximum(25.0)
        self.sigma_spinbox.setValue(1.0)
        self.sigma_spinbox.setSingleStep(0.1)
        layout.addWidget(self.sigma_spinbox)
        layout.addStretch()

    def get_params(self) -> dict:
        return {'sigma': self.sigma_spinbox.value()}

class PowerLawParamsWidget(BaseParamsWidget):
    """A widget for Power Law (Gamma) Transformation."""
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QLabel("Gamma:"))
        self.gamma_spinbox = QDoubleSpinBox()
        self.gamma_spinbox.setMinimum(0.01)
        self.gamma_spinbox.setMaximum(5.0)
        self.gamma_spinbox.setValue(1.0)
        self.gamma_spinbox.setSingleStep(0.1)
        layout.addWidget(self.gamma_spinbox)
        layout.addStretch()

    def get_params(self) -> dict:
        return {'gamma': self.gamma_spinbox.value()}

class ConvolutionParamsWidget(BaseParamsWidget):
    """A widget for defining a 3x3 convolution kernel."""
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QLabel("3x3 Kernel:"))
        
        grid_layout = QGridLayout()
        self.kernel_inputs = []
        for r in range(3):
            row_inputs = []
            for c in range(3):
                spinbox = QDoubleSpinBox()
                spinbox.setMinimum(-100.0)
                spinbox.setMaximum(100.0)
                spinbox.setValue(0.0)
                # Set center to 1.0 for an identity-like default
                if r == 1 and c == 1:
                    spinbox.setValue(1.0)
                grid_layout.addWidget(spinbox, r, c)
                row_inputs.append(spinbox)
            self.kernel_inputs.append(row_inputs)
        layout.addLayout(grid_layout)

    def get_params(self) -> dict:
        kernel = np.array([[spinbox.value() for spinbox in row] for row in self.kernel_inputs])
        return {'kernel': kernel}
    

class BrightnessParamsWidget(BaseParamsWidget):
    """A widget for Brightness Adjustment parameters."""
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(QLabel("Brightness Value (-100 to 100):"))
        self.value_spinbox = QDoubleSpinBox()
        self.value_spinbox.setMinimum(-100.0)
        self.value_spinbox.setMaximum(100.0)
        self.value_spinbox.setValue(20.0)
        layout.addWidget(self.value_spinbox)

        layout.addStretch()

    def get_params(self) -> dict:
        return {"brightness": self.value_spinbox.value()}



# Define a custom control widget
class ReemControlsWidget(QWidget):
    # Signal to request processing from the module manager
    process_requested = Signal(dict)

    def __init__(self, module_manager, parent=None):
        super().__init__(parent)
        self.module_manager = module_manager
        self.param_widgets = {}
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("<h3>Control Panel</h3>"))

        layout.addWidget(QLabel("Operation:"))
        self.operation_selector = QComboBox()
        layout.addWidget(self.operation_selector)

        # Stacked widget to hold the parameter UIs
        self.params_stack = QStackedWidget()
        layout.addWidget(self.params_stack)

        # Define operations and their corresponding parameter widgets
        operations = {
            "Vintage Filter": NoParamsWidget,
            "Brightness Adjustment": BrightnessParamsWidget,
            "Sharpening": NoParamsWidget,
            "Gaussian Blur": GaussianParamsWidget,
            "Sobel Edge Detect": NoParamsWidget,
            "Power Law (Gamma)": PowerLawParamsWidget,
            "Convolution": ConvolutionParamsWidget,
        }

        for name, widget_class in operations.items():
            widget = widget_class()
            self.params_stack.addWidget(widget)
            self.param_widgets[name] = widget
            self.operation_selector.addItem(name)

        self.apply_button = QPushButton("Apply Processing")
        layout.addWidget(self.apply_button)

        self.apply_button.clicked.connect(self._on_apply_clicked)
        self.operation_selector.currentTextChanged.connect(self._on_operation_changed)

    def _on_apply_clicked(self):
        operation_name = self.operation_selector.currentText()
        active_widget = self.param_widgets[operation_name]
        params = active_widget.get_params()
        params['operation'] = operation_name # Add operation name to params
        self.process_requested.emit(params)

    def _on_operation_changed(self, operation_name: str):
        if operation_name in self.param_widgets:
            self.params_stack.setCurrentWidget(self.param_widgets[operation_name])

class ReemImageModule(IImageModule):
    def __init__(self):
        super().__init__()
        self._controls_widget = None

    def get_name(self) -> str:
        return "Reem Image Enhancement Module"

    def get_supported_formats(self) -> list[str]:
        return ["png", "jpg", "jpeg", "bmp", "gif", "tiff"] # Common formats

    def create_control_widget(self, parent=None, module_manager=None) -> QWidget:
        if self._controls_widget is None:
            self._controls_widget = ReemControlsWidget(module_manager, parent)
            # The widget's signal is connected to the module's handler
            self._controls_widget.process_requested.connect(self._handle_processing_request)
        return self._controls_widget

    def _handle_processing_request(self, params: dict):
        # Here, the module needs a way to trigger processing in the main app
        # The control widget now has a valid reference to the module manager
        if self._controls_widget and self._controls_widget.module_manager:
            self._controls_widget.module_manager.apply_processing_to_current_image(params)

    def load_image(self, file_path: str):
        try:
            image_data = imageio.imread(file_path)

            if image_data.ndim == 3 and image_data.shape[2] in [3, 4]:
                pass
            elif image_data.ndim == 2:
                image_data = image_data[np.newaxis, :]
            else:
                print(f"Warning: Unexpected image dimensions {image_data.shape}")

            metadata = {"name": file_path.split("/")[-1]}
            return True, image_data, metadata, None

        except Exception as e:
            print(f"Error loading 2D image {file_path}: {e}")
            return False, None, {}, None

    def process_image(self, image_data: np.ndarray, metadata: dict, params: dict) -> np.ndarray:
        processed_data = image_data.copy()
        operation = params.get("operation")

        if operation == "Vintage Filter":
            img = processed_data.astype(np.float32)

            if img.ndim == 3 and img.shape[2] >= 3:
                img[:, :, 0] *= 1.10
                img[:, :, 1] *= 1.03
                img[:, :, 2] *= 0.88

                img[:, :, :3] = 128 + 1.08 * (img[:, :, :3] - 128)
                img[:, :, :3] = img[:, :, :3] * 0.95 + 12

            processed_data = np.clip(img, 0, 255)

        elif operation == "Brightness Adjustment":
            value = params.get("brightness", 20.0)

            img = processed_data.astype(np.float32)
            processed_data = img + value
            processed_data = np.clip(processed_data, 0, 255)

        elif operation == "Sharpening":
            kernel = np.array([
                [0, -1, 0],
                [-1, 5, -1],
                [0, -1, 0]
            ])

            img = processed_data.astype(float)

            if img.ndim == 3 and img.shape[2] in [3, 4]:
                channels = []
                for i in range(img.shape[2]):
                    channels.append(convolve(img[:, :, i], kernel, mode="reflect"))
                processed_data = np.stack(channels, axis=-1)
            else:
                processed_data = convolve(img, kernel, mode="reflect")

        elif operation == "Gaussian Blur":
            sigma = params.get("sigma", 1.0)

            processed_data = skimage.filters.gaussian(
                processed_data.astype(float),
                sigma=sigma,
                preserve_range=True
            )

        elif operation == "Median Filter":
            filter_size = params.get("filter_size", 3)

            if filter_size > 1:
                if processed_data.ndim == 3 and processed_data.shape[2] in [3, 4]:
                    channels = []

                    for i in range(processed_data.shape[2]):
                        channels.append(
                            skimage.filters.median(
                                processed_data[:, :, i],
                                footprint=skimage.morphology.disk(int(filter_size / 2))
                            )
                        )

                    processed_data = np.stack(channels, axis=-1)

                else:
                    processed_data = skimage.filters.median(
                        processed_data,
                        footprint=skimage.morphology.disk(int(filter_size / 2))
                    )

        elif operation == "Sobel Edge Detect":
            if processed_data.ndim == 3 and processed_data.shape[2] in [3, 4]:
                grayscale_img = rgb2gray(processed_data[:, :, :3])
            else:
                grayscale_img = processed_data

            processed_data = skimage.filters.sobel(grayscale_img)

        elif operation == "Power Law (Gamma)":
            gamma = params.get("gamma", 1.0)

            input_float = processed_data.astype(float)
            max_val = np.max(input_float)

            if max_val > 0:
                normalized = input_float / max_val
                gamma_corrected = np.power(normalized, gamma)
                processed_data = gamma_corrected * max_val

        elif operation == "Convolution":
            kernel = params.get("kernel")

            if kernel is not None:
                input_float = processed_data.astype(float)

                if input_float.ndim == 3 and input_float.shape[2] in [3, 4]:
                    channels = []

                    for i in range(input_float.shape[2]):
                        channels.append(
                            convolve(input_float[:, :, i], kernel, mode="reflect")
                        )

                    processed_data = np.stack(channels, axis=-1)

                else:
                    processed_data = convolve(input_float, kernel, mode="reflect")

        processed_data = np.clip(processed_data, 0, 255)
        processed_data = processed_data.astype(image_data.dtype)

        return processed_data
