# Python Image Processing Toolkit

A comprehensive Python toolkit for fundamental image processing operations using OpenCV and NumPy.

## Features

- **Intensity Level Reduction**: Reduce image intensity levels to any power of 2 (2, 4, 8, 16, etc.)
- **Spatial Averaging**: Apply neighborhood averaging with customizable kernel sizes (3x3, 10x10, 20x20)
- **Image Rotation**: Rotate images by any angle (45°, 90°, or custom angles)
- **Block Averaging**: Reduce spatial resolution using non-overlapping block averaging (3x3, 5x5, 7x7)

## Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Required Libraries

```bash
pip install opencv-python numpy matplotlib
```

## Usage

### Basic Usage

```python
from image_processor import ImageProcessor

# Initialize with your image
processor = ImageProcessor("path/to/your/image.jpg")

# Reduce intensity levels
reduced_image = processor.reduce_intensity_levels(16)

# Apply spatial averaging
averaged_image = processor.spatial_averaging(3)

# Rotate image
rotated_image = processor.rotate_image(45)

# Apply block averaging
block_averaged = processor.block_averaging(5)
```

### Running the Complete Demo

```bash
python image_processing.py
```

Make sure to update the `image_path` variable in the `main()` function with your image file path.

## API Reference

### ImageProcessor Class

#### `__init__(image_path: str)`
Initialize the processor with an image file.

#### `reduce_intensity_levels(num_levels: int) -> np.ndarray`
Reduce the number of intensity levels in the image.
- **Parameters**: `num_levels` - Must be a power of 2 (2, 4, 8, 16, 32, 64, 128, 256)
- **Returns**: Image with reduced intensity levels

#### `spatial_averaging(kernel_size: int) -> np.ndarray`
Apply spatial averaging with specified kernel size.
- **Parameters**: `kernel_size` - Size of the averaging kernel (e.g., 3 for 3x3)
- **Returns**: Spatially averaged image

#### `rotate_image(angle: float) -> np.ndarray`
Rotate the image by specified angle.
- **Parameters**: `angle` - Rotation angle in degrees
- **Returns**: Rotated image (dimensions adjusted to prevent cropping)

#### `block_averaging(block_size: int) -> np.ndarray`
Replace non-overlapping blocks with their average values.
- **Parameters**: `block_size` - Size of blocks (e.g., 3 for 3x3 blocks)
- **Returns**: Block-averaged image with reduced spatial resolution

#### `display_results(images: list, titles: list, figsize: tuple)`
Display multiple images in a grid layout.

## Examples

### Intensity Level Reduction

```python
# Reduce to 2, 4, and 16 intensity levels
processor = ImageProcessor("sample.jpg")
img_2_levels = processor.reduce_intensity_levels(2)
img_4_levels = processor.reduce_intensity_levels(4)
img_16_levels = processor.reduce_intensity_levels(16)
```

### Spatial Averaging

```python
# Apply different kernel sizes
avg_3x3 = processor.spatial_averaging(3)
avg_10x10 = processor.spatial_averaging(10)
avg_20x20 = processor.spatial_averaging(20)
```

### Image Rotation

```python
# Rotate by different angles
rotated_45 = processor.rotate_image(45)
rotated_90 = processor.rotate_image(90)
rotated_custom = processor.rotate_image(30)
```

### Block Averaging

```python
# Different block sizes
block_3x3 = processor.block_averaging(3)
block_5x5 = processor.block_averaging(5)
block_7x7 = processor.block_averaging(7)
```

## Supported Image Formats

- JPEG (.jpg, .jpeg)
- PNG (.png)
- BMP (.bmp)
- TIFF (.tiff, .tif)
- And other formats supported by OpenCV

## Algorithm Details

### Intensity Level Reduction
Uses bit manipulation to efficiently reduce intensity levels:
```python
shift_amount = 8 - log2(num_levels)
reduced_image = (image >> shift_amount) << shift_amount
```

### Spatial Averaging
Applies convolution with uniform averaging kernels:
```python
kernel = np.ones((size, size), np.float32) / (size * size)
result = cv2.filter2D(image, -1, kernel)
```

### Image Rotation
Uses affine transformation to rotate the given image at the given angle.

### Block Averaging
Reshapes image into blocks and computes average values:
```python
reshaped = image.reshape(h//block_size, block_size, w//block_size, block_size)
averages = np.mean(reshaped, axis=(1, 3))
```


