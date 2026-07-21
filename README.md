# Imaging Technologies Capstone Project: SRH ImageViewer

Welcome to the SRH Image Processing Capstone project! The goal of this project is to collaboratively build an image processing application. Each student will contribute by creating a personal "module" that adds new image processing capabilities to the main application.

![alt text](./assets/app.png)

This guide will walk you through setting up the project, creating your own module, and submitting your contribution.

## Project Timeline

-   **Project Start:** Now
-   **Project Deadline:** All pull requests must be created and submitted by **Wednesday, 22nd July at midnight**.
-   **Project Presentations:** Thursday 23rd

## 1. Project Setup

First, you need to get the project code running on your local machine.

### 1.1. Clone the Repository

Open your terminal or Git client and clone the project repository to your computer.

```bash
git clone https://github.com/guijoe/srh-image-processing-app.git
cd srh-image-processing-app
```

### 1.2. Create a Virtual Environment

You can choose to install the project's dependencies in your global python environment. If that is the case, please skip to 1.3.

However, it might be more adequate to use a virtual environment to manage project dependencies. This keeps your project's libraries separate from your system's Python installation.

```bash
# Create a virtual environment named 'venv'
python -m venv venv

# Activate the environment
# On Windows:
.\venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

You will see `(venv)` at the beginning of your terminal prompt, indicating the environment is active.

### 1.3. Install Dependencies

Install all the necessary Python libraries using the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

or

```bash
python -m pip install -r requirements.txt
```

You should now be able to run the main application.

```bash
cd src
python main_app.py
```

If the app does not run, check the error that is displayed. It is possible that you need to install `PyQt6`. Un-comment on the first line of your `requirements.txt` file and re-run the dependencies installation comment. Re-try running the app.

## 2. Creating Your Module

Your contribution will be a new "module". A module is a self-contained set of files that adds your custom functionality (a set of image processing features or transformations) and user interface (UI) to the main application. You can add as many features as you like.

### 2.1. Create Your Module Directory and File

1.  Navigate to the `modules/` directory.
2.  Create a new folder with your name (e.g., `john_doe`).
3.  Inside your new folder, create a Python file. This file name should contain your module's name (e.g. `john_doe`), plus the suffix `_module`. For example `john_doe_module.py`.

Your project structure should look like this:

```
srhimageviewer_test/
├── modules/
│   ├── sample/
│   │   └── sample_module.py
│   ├── john_doe/
│   │   └── john_doe_module.py  <-- Your new file
│   └── i_image_module.py
├── main.py
└── requirements.txt
```

### 2.2. Use the Template

To get started quickly, copy the contents of an existing module like `modules/sample/sample_module.py` into your new `john_doe_module.py` file. You will then modify this template for your own transformation. Whereever the word `Sample` appears in your new file, you can replace it with your module's name (`e.g. JohnDoe`)

## 3. Adding a Custom Transformation

Let's add a "Contrast Stretching" transformation. This is a simple point operation that enhances the contrast of an image by stretching the range of intensity values. It's a great example because it requires user-defined parameters.

### 3.1. Step 1: Create the UI for Parameters

First, we need to create the UI widgets that will allow the user to input the parameters for our transformation. In your module file (`john_doe_module.py`), find the section with parameter widgets and add a new class for contrast stretching.

This widget will have two spin boxes for the user to select the desired minimum and maximum output intensity values.

```python
# In your_module.py, add this class near the other *ParamsWidget classes

class ContrastStretchingParamsWidget(BaseParamsWidget):
    """A widget for Contrast Stretching parameters."""
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Input for the new minimum value
        layout.addWidget(QLabel("New Minimum Intensity (0-255):"))
        self.min_spinbox = QDoubleSpinBox()
        self.min_spinbox.setMinimum(0.0)
        self.min_spinbox.setMaximum(255.0)
        self.min_spinbox.setValue(0.0)
        layout.addWidget(self.min_spinbox)

        # Input for the new maximum value
        layout.addWidget(QLabel("New Maximum Intensity (0-255):"))
        self.max_spinbox = QDoubleSpinBox()
        self.max_spinbox.setMinimum(0.0)
        self.max_spinbox.setMaximum(255.0)
        self.max_spinbox.setValue(255.0)
        layout.addWidget(self.max_spinbox)

        layout.addStretch()

    def get_params(self) -> dict:
        return {
            'new_min': self.min_spinbox.value(),
            'new_max': self.max_spinbox.value()
        }
```

### 3.2. Step 2: Add the Operation to the Control Panel

Now, connect your new parameter widget to the dropdown menu in your module's control panel.

In your `YourNameControlsWidget` class (e.g., `JohnDoeControlsWidget`), find the `operations` dictionary and add an entry for "Contrast Stretching" (You can remove all transformations that you do not wish to implement).

```python
# In your YourNameControlsWidget class, inside the setup_ui method

        # ...
        # Define operations and their corresponding parameter widgets
        operations = {
            "Gaussian Blur": GaussianParamsWidget,
            "Sobel Edge Detect": NoParamsWidget,
            "Power Law (Gamma)": PowerLawParamsWidget,
            "Convolution": ConvolutionParamsWidget,
            "Contrast Stretching": ContrastStretchingParamsWidget, # <-- Add this line
        }
        # ...
```

### 3.3. Step 3: Implement the Image Processing Logic

This is where you write the core NumPy code for the transformation. In your main module class (e.g., `JohnDoeImageModule`), find the `process_image` method and add the logic for your new operation. You can choose to implement this logic in a separate function and just call the function in the `process_image` method.

The logic is as follows:
1.  Get the current minimum and maximum intensity of the input image.
2.  Get the desired new minimum and maximum from the user parameters.
3.  Apply the linear scaling formula to each pixel.
4.  Ensure the output data type is the same as the input.

```python
# In your YourNameImageModule class, inside the process_image method

    def process_image(self, image_data: np.ndarray, metadata: dict, params: dict) -> np.ndarray:
        processed_data = image_data.copy()
        operation = params.get('operation')

        # ... (keep the other elif blocks for other operations if necessary)

        elif operation == "Contrast Stretching":
            # Ensure we are working with a floating point image for calculations
            img_float = processed_data.astype(float)

            # Get parameters from the UI
            new_min = params.get('new_min', 0.0)
            new_max = params.get('new_max', 255.0)

            # Get current image intensity range
            current_min = np.min(img_float)
            current_max = np.max(img_float)

            # Avoid division by zero if the image is flat
            if current_max == current_min:
                return processed_data # Return original image

            # Apply the linear stretching formula
            processed_data = (img_float - current_min) * \
                             ((new_max - new_min) / (current_max - current_min)) + new_min

            # Clip values to be safe, though the formula should handle it
            processed_data = np.clip(processed_data, new_min, new_max)

        # ... (keep the rest of the function)

        # Ensure output data type is consistent
        processed_data = processed_data.astype(image_data.dtype)

        return processed_data
```

Now, run the application. You should see your module in the module selection dropdown, and "Contrast Stretching" should appear in its operations list!

## 4. Submitting Your Contribution

Once you have tested and are happy with all functionalities implementated in your module, it is time to submit it for review (latest 17th for Group 1 and 18th for Group 2).

### 4.1. Use Git for Version Control

When you are done writing and testing your code:

1.  **Create a new branch** for your feature. This keeps your work separate from the main codebase.

    ```bash
    # Create and switch to a new branch
    git checkout -b feature/john-doe
    ```

2.  **Commit your changes.** Add the files you've changed and create a commit with a descriptive message.

    ```bash
    # Add your new module file
    git add src/modules/john_doe/john_doe_module.py

    # Commit the changes
    git commit -m "feat: Add John Doe module"
    ```

3.  **Push your branch** to the remote repository.

    ```bash
    git push -u origin feature/john-doe
    ```

### 4.2. Create a Pull Request

After executing the last command, follow the link and instructions on your terminal to create a pull request.

Your pull request will be reviewed and, once approved, merged into the main project. Congratulations, you've successfully contributed and submitted your work to be graded ! See you the next for presentations !
