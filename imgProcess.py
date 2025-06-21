import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

class ImageProcessor:
    
    # Load the image from the specified path
    def __init__(self, image_path: str):
        self.original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if self.original_image is None:
            raise ValueError(f"Could not load image from {image_path}")
        print(f"Loaded image with shape: {self.original_image.shape}")
    
    def reduce_intensity_levels(self, num_levels: int) -> np.ndarray:
    
        # Calculate the number of bits needed
        bits = int(np.log2(num_levels))
        
        # Right shift to reduce bits, then left shift to restore scale
        shift_amount = 8 - bits
        reduced_image = (self.original_image >> shift_amount) << shift_amount
        
        return reduced_image
    
    def spatial_averaging(self, kernel_size: int) -> np.ndarray:

        # Create averaging kernel
        kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
        
        # Apply convolution
        averaged_image = cv2.filter2D(self.original_image, -1, kernel)
        
        return averaged_image.astype(np.uint8)
    
    def rotate_image(self, angle: float) -> np.ndarray:

        height, width = self.original_image.shape
        center = (width // 2, height // 2)
        
        # Get rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Apply rotation
        rotated_image = cv2.warpAffine(self.original_image, rotation_matrix, (width, height))
        
        return rotated_image
    
    def block_averaging(self, block_size: int) -> np.ndarray:

        height, width = self.original_image.shape
        
        # Calculate new dimensions
        new_height = (height // block_size) * block_size
        new_width = (width // block_size) * block_size
        
        # Crop image to fit exact blocks
        cropped_image = self.original_image[:new_height, :new_width]
        
        # Reshape to separate blocks
        reshaped = cropped_image.reshape(
            new_height // block_size, block_size,
            new_width // block_size, block_size
        )
        
        # Calculate average for each block
        block_averages = np.mean(reshaped, axis=(1, 3))
        
        # Expand averages back to original block size
        expanded_averages = np.repeat(np.repeat(block_averages, block_size, axis=0), block_size, axis=1)
        
        return expanded_averages.astype(np.uint8)
    
    def display_results(self, images: list, titles: list, window_title:str, figsize: Tuple[int, int] = (15, 10)):

        n_images = len(images)
        cols = 3
        rows = (n_images + cols - 1) // cols
        
        fig = plt.figure(figsize=figsize)

        fig.suptitle(window_title, fontsize=16, fontweight='bold')
        
        for i, (img, title) in enumerate(zip(images, titles)):
            plt.subplot(rows, cols, i + 1)
            plt.imshow(img, cmap='gray')
            plt.title(title)
            plt.axis('off')
        
        plt.tight_layout()
        plt.show()


def main():
    
    image_path = "img.jpg"

    processor = ImageProcessor(image_path)
    
    # 1. Intensity Level Reduction
    print("1. Performing intensity level reduction...")
    intensity_2 = processor.reduce_intensity_levels(2)
    intensity_4 = processor.reduce_intensity_levels(4)
    intensity_16 = processor.reduce_intensity_levels(16)
    
    # Display intensity reduction results
    processor.display_results(
        [processor.original_image, intensity_2, intensity_4, intensity_16],
        ['Original (256 levels)', '2 levels', '4 levels', '16 levels'],
         "Intensity Level Reduction"
    )
    
    # 2. Spatial Averaging
    print("2. Performing spatial averaging...")
    avg_3x3 = processor.spatial_averaging(3)
    avg_10x10 = processor.spatial_averaging(10)
    avg_20x20 = processor.spatial_averaging(20)
    
    # Display spatial averaging results
    processor.display_results(
        [processor.original_image, avg_3x3, avg_10x10, avg_20x20],
        ['Original', '3x3 Average', '10x10 Average', '20x20 Average'],
        "Spatial Averaging Operations"
    )
    
    # 3. Image Rotation
    print("3. Performing image rotation...")
    rotated_45 = processor.rotate_image(45)
    rotated_90 = processor.rotate_image(90)
    
    # Display rotation results
    processor.display_results(
        [processor.original_image, rotated_45, rotated_90],
        ['Original', '45° Rotation', '90° Rotation'],
        "Image Rotation Operations"
    )
    
    # 4. Block Averaging (Spatial Resolution Reduction)
    print("4. Performing block averaging...")
    block_3x3 = processor.block_averaging(3)
    block_5x5 = processor.block_averaging(5)
    block_7x7 = processor.block_averaging(7)
    
    # Display block averaging results
    processor.display_results(
        [processor.original_image, block_3x3, block_5x5, block_7x7],
        ['Original', '3x3 Blocks', '5x5 Blocks', '7x7 Blocks'],
        "Block Averaging (Resolution Reduction)"
    )

if __name__ == "__main__":
    main()