# JPEG Compression Tool

## Overview
This JPEG Compression Tool is an educational application that demonstrates the JPEG compression algorithm step-by-step. It provides a visual interface to observe how JPEG compression works, from color space conversion to discrete cosine transform (DCT) and quantization.

## Features
- **Image Selection**: Load any image for compression analysis
- **Quality Control**: Adjust compression quality with an intuitive slider
- **Visualization Tabs**:
  - Original image display
  - YCbCr color space components (luminance and chrominance)
  - DCT coefficients visualization
  - Quantized DCT coefficients
  - Reconstructed image components
  - Side-by-side comparison between original and compressed images
- **PSNR Calculation**: Measure quality degradation with Peak Signal-to-Noise Ratio

## Technical Background
The application implements the core steps of JPEG compression:

1. **Color Space Conversion**: Converts RGB to YCbCr separating luminance (Y) from chrominance (Cb, Cr)
2. **Downsampling**: Reduces chrominance resolution (4:2:0 subsampling)
3. **Block Splitting**: Divides the image into 8×8 pixel blocks
4. **Discrete Cosine Transform (DCT)**: Transforms spatial data into frequency components
5. **Quantization**: Reduces precision of frequency components based on quality setting
6. **Dequantization & IDCT**: Reconstructs the image for display
7. **Quality Measurement**: Calculates PSNR to quantify compression loss

## Usage
1. Launch the application
2. Click "Browse Image" to select an image file
3. Adjust the quality slider (0 = best quality, 50 = worst quality)
4. Click "Process Image" to start compression
5. Navigate through tabs to explore different stages of compression
6. The status bar displays PSNR and processing information

## Dependencies
- Python 3.x
- CustomTkinter
- NumPy
- Matplotlib
- Pillow (PIL)

## Installation
```bash
pip install -h requirements.txt
```

## Running the Application
```bash
python tpm.py
```

## Educational Value
This tool visualizes how lossy compression affects image data, making it valuable for:
- Students learning digital signal processing
- Computer graphics courses
- Image compression demonstrations
- Understanding the tradeoffs between file size and image quality

The quality slider enables experimentation with different compression levels, showing how quantization affects both visual quality and theoretical measurements like PSNR.

## License
[MIT License](LICENSE)