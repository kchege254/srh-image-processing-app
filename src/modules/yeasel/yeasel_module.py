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

# Define a custom control widget
class YeaselControlsWidget(QWidget):
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
            "Contrast Stretching": NoParamsWidget,
            "Power Law (Gamma)": PowerLawParamsWidget,
            "Image Negative": NoParamsWidget,
            "Sharpen Image": NoParamsWidget,
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

class YeaselImageModule(IImageModule):
    def __init__(self):
        super().__init__()
        self._controls_widget = None

    def get_name(self) -> str:
        return "Yeasel Image Processing Module"

    def get_supported_formats(self) -> list[str]:
        return ["png", "jpg", "jpeg", "bmp", "gif", "tiff"] # Common formats

    def create_control_widget(self, parent=None, module_manager=None) -> QWidget:
        if self._controls_widget is None:
            self._controls_widget = YeaselControlsWidget(module_manager, parent)
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
            # Ensure 2D images are correctly shaped (e.g., handle grayscale vs RGB)
            if image_data.ndim == 3 and image_data.shape[2] in [3, 4]: # RGB or RGBA
                # napari handles this well, but for processing, sometimes a single channel is needed
                pass
            elif image_data.ndim == 2: # Grayscale
                image_data = image_data[np.newaxis, :] # Add a channel dimension for consistency if desired
            else:
                print(f"Warning: Unexpected image dimensions {image_data.shape}")

            metadata = {'name': file_path.split('/')[-1]}
            # Add more metadata: original_shape, file_size, etc.
            return True, image_data, metadata, None # Session ID generated by store
        except Exception as e:
            print(f"Error loading 2D image {file_path}: {e}")
            return False, None, {}, None
    
    def process_image(
    self,
    image_data: np.ndarray,
    metadata: dict,
    params: dict
) -> np.ndarray:

        operation = params.get("operation")
        input_float = image_data.astype(np.float32)
        processed_data = input_float.copy()

        if operation == "Contrast Stretching":
            # Preserve the alpha channel when processing RGBA images.
            if input_float.ndim == 3 and input_float.shape[2] == 4:
                image_channels = input_float[:, :, :3]
                alpha_channel = input_float[:, :, 3:4]

                min_value = np.min(image_channels)
                max_value = np.max(image_channels)

                if max_value > min_value:
                    stretched = (
                        (image_channels - min_value)
                        / (max_value - min_value)
                        * 255.0
                    )
                else:
                    stretched = image_channels.copy()

                processed_data = np.concatenate(
                    (stretched, alpha_channel),
                    axis=2
                )

            else:
                min_value = np.min(input_float)
                max_value = np.max(input_float)

                if max_value > min_value:
                    processed_data = (
                        (input_float - min_value)
                        / (max_value - min_value)
                        * 255.0
                    )
                else:
                    processed_data = input_float.copy()

        elif operation == "Power Law (Gamma)":
            gamma = params.get("gamma", 1.0)

            if input_float.ndim == 3 and input_float.shape[2] == 4:
                rgb = input_float[:, :, :3] / 255.0
                alpha = input_float[:, :, 3:4]

                corrected_rgb = np.power(rgb, gamma) * 255.0
                processed_data = np.concatenate(
                    (corrected_rgb, alpha),
                    axis=2
                )
            else:
                normalized = input_float / 255.0
                processed_data = np.power(normalized, gamma) * 255.0

        elif operation == "Image Negative":
            if input_float.ndim == 3 and input_float.shape[2] == 4:
                processed_data[:, :, :3] = 255.0 - input_float[:, :, :3]
                processed_data[:, :, 3] = input_float[:, :, 3]
            else:
                processed_data = 255.0 - input_float

        elif operation == "Sharpen Image":
            sharpen_kernel = np.array(
                [
                    [0, -2, 0],
                    [-2, 13, -2],
                    [0, -2, 0]
                ],
                dtype=np.float32
            )

            def apply_convolution(channel: np.ndarray) -> np.ndarray:
                padded = np.pad(channel, 1, mode="reflect")
                result = np.zeros_like(channel, dtype=np.float32)

                for row in range(channel.shape[0]):
                    for column in range(channel.shape[1]):
                        region = padded[row:row + 3, column:column + 3]
                        result[row, column] = np.sum(
                            region * sharpen_kernel
                        )

                return result

            if input_float.ndim == 2:
                processed_data = apply_convolution(input_float)

            elif input_float.ndim == 3:
                processed_data = input_float.copy()
                number_of_colour_channels = min(input_float.shape[2], 3)

                for channel_index in range(number_of_colour_channels):
                    processed_data[:, :, channel_index] = apply_convolution(
                        input_float[:, :, channel_index]
                    )

                # Preserve alpha in RGBA images.
                if input_float.shape[2] == 4:
                    processed_data[:, :, 3] = input_float[:, :, 3]

        # Prevent invalid values and restore the original datatype.
        processed_data = np.clip(processed_data, 0, 255)
        return processed_data.astype(image_data.dtype)