import numpy as np
from PIL import Image

def get_brenners_focus_horizontal_derivative(pil_image: Image.Image) -> float:
    """
    Calculate Brenner's focus measure to assess the sharpness of a given image.
    Variation 1 of Brenner's focus measure

    Args:
        pil_image (Image.Image): A PIL Image object to be analyzed.

    Returns:
        float: Brenner's focus measure for the image.
    """
    
    # Ensure the input image is in grayscale mode ('L' mode)
    if pil_image.mode != 'L':
        pil_image = pil_image.convert('L')

    # Convert the PIL image to a NumPy array for efficient processing
    image = np.array(pil_image)

    # Compute the difference between adjacent pixels along the vertical axis (axis=0)
    diff = np.diff(image, axis=1)

    # Calculate Brenner's focus measure by summing the squares of the differences
    focus_measure = np.sum(diff**2)

    return focus_measure


def get_brenners_focus_vert_derivative(pil_image: Image.Image) -> float:
    """
    Calculate Brenner's focus measure to assess the sharpness of a given image.
    Variation 2 of Brenner's focus measure
    Args:
        pil_image (Image.Image): A PIL Image object to be analyzed.

    Returns:
        float: Brenner's focus measure for the image.
    """
    
    # Ensure the input image is in grayscale mode ('L' mode)
    if pil_image.mode != 'L':
        pil_image = pil_image.convert('L')

    # Convert the PIL image to a NumPy array for efficient processing
    image = np.array(pil_image)

    # Compute the difference between adjacent pixels along the vertical axis (axis=0)
    diff = np.diff(image, axis=0)

    # Calculate Brenner's focus measure by summing the squares of the differences
    focus_measure = np.sum(diff**2)

    return focus_measure


def get_brenners_focus_horizontal_gradient(pil_image: Image.Image) -> float:
    """
    Calculate Brenner's focus measure to assess the sharpness of a given image.
    Variation 3 of Brenner's focus measure

    Args:
        pil_image (Image.Image): A PIL Image object to be analyzed.

    Returns:
        float: Brenner's focus measure for the image.
    """
    
    # Ensure the input image is in grayscale mode ('L' mode)
    if pil_image.mode != 'L':
        pil_image = pil_image.convert('L')

    # Convert the PIL image to a NumPy array for efficient processing
    image = np.array(pil_image)

    # Gradient in the horizontal direction
    grad_x = np.gradient(image, axis=1)  

    # Sum of squares of horizontal gradient
    focus_measure = np.sum(grad_x**2)

    return focus_measure


def get_brenners_focus_vert_gradient(pil_image: Image.Image) -> float:
    """
    Calculate Brenner's focus measure to assess the sharpness of a given image.
    Variation 4 of Brenner's focus measure

    Args:
        pil_image (Image.Image): A PIL Image object to be analyzed.

    Returns:
        float: Brenner's focus measure for the image.
    """
    
    # Ensure the input image is in grayscale mode ('L' mode)
    if pil_image.mode != 'L':
        pil_image = pil_image.convert('L')

    # Convert the PIL image to a NumPy array for efficient processing
    image = np.array(pil_image)

    # Gradient in the vertical direction
    grad_x = np.gradient(image, axis=0)  

    # Sum of squares of vertical gradient
    focus_measure = np.sum(grad_x**2)

    return focus_measure


def get_brenners_focus_color(pil_image: Image.Image) -> float:
    """
    Calculate Brenner's focus measure for a color image by applying the measure
    to each color channel (R, G, B) and then averaging the focus measure values.

    Args:
        pil_image (Image.Image): A PIL Image object to be analyzed.

    Returns:
        float: Brenner's focus measure for the image.
    """
    
    # Convert the PIL image to a NumPy array
    image = np.array(pil_image)
    
    # Split the image into its R, G, B channels
    if image.ndim == 3:  # Check if it's a color image
        red_channel = image[:, :, 0]
        green_channel = image[:, :, 1]
        blue_channel = image[:, :, 2]
    else:
        raise ValueError("Input image must be a color image with 3 channels.")

    # Define a helper function to compute Brenner's focus measure for a single channel
    def brenner_focus_single_channel(channel):
        grad_x = np.roll(channel, -2, axis=1) - channel  # Horizontal shift by 2 pixels
        focus_measure = np.sum(grad_x[:-2, :] ** 2)
        return focus_measure

    # Compute Brenner's focus measure for each channel
    red_focus = brenner_focus_single_channel(red_channel)
    green_focus = brenner_focus_single_channel(green_channel)
    blue_focus = brenner_focus_single_channel(blue_channel)

    # Aggregate the focus measures (average, sum, or max)
    focus_measure = (red_focus + green_focus + blue_focus) / 3  # Average the focus values

    return focus_measure