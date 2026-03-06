# LEGO Brick Classifier

## Project Overview

To run the code, execute **`main.py`**.

---

### File Descriptions

- **`mapping.py`**  
  Contains dictionaries that map each piece and color ID to a unique integer.

- **`transformations.py`**  
  Defines functions to:
  - Resize input images to **64x64**
  - Normalize the images
  - Convert them into **PyTorch tensors**  
  These transformations ensure compatibility with the convolutional neural network (CNN) models.

- **`classifier_cnn.py`**  
  Contains the definition of the CNN model used to classify **piece IDs** and **color IDs**.

- **`classifier.py`**  
  Defines a function that:
  1. Creates an instance of the model from `classifier_cnn.py`
  2. Applies the image transformation functions from `transformations.py`
  3. Feeds the processed image into the model
  4. Returns the predicted **piece** and **color** labels

- **`is_lego_cnn.py`**  
  Contains the definition of the CNN model used to determine whether an image contains a LEGO brick.

- **`is_lego.py`**  
  Defines a function that:
  1. Creates an instance of the model from `is_lego_cnn.py`
  2. Applies the transformation functions from `transformations.py`
  3. Feeds the processed image into the model
  4. Returns a label indicating whether a LEGO brick is present

- **`gui.py`**  
  Implements a **Gradio web interface** that allows users to:
  1. Upload an image
  2. Use the `is_lego` function to detect the presence of a LEGO brick
  3. If detected, use the `classifier` function to classify the brick’s **piece** and **color**

- **`weights/`**  
  Contains the `.pth` files used to load pretrained weights for the models.
