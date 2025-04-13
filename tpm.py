import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image
import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog

def rgb_to_ycbcr(image):

    transform_matrix = np.array([
        [0.299, 0.587, 0.114],
        [-0.168736, -0.331264, 0.5],
        [0.5, -0.418688, -0.081312]
    ])

    img_array = np.array(image, dtype=np.float32)
    R, G, B = img_array[:, :, 0], img_array[:, :, 1], img_array[:, :, 2]

    Y = transform_matrix[0, 0] * R + transform_matrix[0, 1] * G + transform_matrix[0, 2] * B
    Cb = transform_matrix[1, 0] * R + transform_matrix[1, 1] * G + transform_matrix[1, 2] * B + 128
    Cr = transform_matrix[2, 0] * R + transform_matrix[2, 1] * G + transform_matrix[2, 2] * B + 128

    return Y, Cb, Cr



def ycbcr_to_rgb(Y, Cb, Cr):
    """Convert YCbCr back to RGB color space"""

    # Upsample Cb and Cr to match Y's dimensions
    Cb , Cr = upsample(Cb, Cr, "3")

    # Assume Cb and Cr are in [0, 255] and need to be centered
    Cb -= 128.0
    Cr -= 128.0

    inv_transform = np.array([
            [1, 0, 1.402],
            [1, -0.344136, -0.714136],
            [1, 1.772, 0]
        ])

    R = inv_transform[0, 0] * Y + inv_transform[0, 2] * Cr
    G = inv_transform[1, 0] * Y + inv_transform[1, 1] * Cb + inv_transform[1, 2] * Cr
    B = inv_transform[2, 0] * Y + inv_transform[2, 1] * Cb
    rgb_image = np.clip(np.stack((R, G, B), axis=-1), 0, 255).astype(np.uint8)
    return rgb_image

    
def psnr(original, compressed):
        """Calculate Peak Signal-to-Noise Ratio"""
        mse = np.mean((original.astype(np.float32) - compressed.astype(np.float32)) ** 2)
        return 10 * np.log10(255 ** 2 / mse) if mse != 0 else float('inf')

 
def blockwise_transform(img, block_size, transform_func, D=None): # to apply DCT or IDCT where needed, used to extract blocks
    """Apply a transformation blockwise to the image"""
    h, w = img.shape
    transformed_img = np.zeros_like(img, dtype=np.float32)

    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            if D is not None:
                transformed_img[i:i + block_size, j:j + block_size] = transform_func(
                    img[i:i + block_size, j:j + block_size], D)
            else:
                transformed_img[i:i + block_size, j:j + block_size] = transform_func(
                    img[i:i + block_size, j:j + block_size])

    return transformed_img

def rescale_idct_output(component, original_min, original_max):
    """Rescale IDCT output to match expected YCbCr ranges"""
 
    if original_max == original_min:
        return np.ones_like(component) * original_min
    
    # Scale to original range
    return (component - np.min(component)) / (np.max(component) - np.min(component)) * \
           (original_max - original_min) + original_min


def dct_matrix(N):
    D = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i == 0:
                D[i, j] = np.sqrt(1 / N)
            else:
                D[i, j] = np.sqrt(2 / N) * np.cos((np.pi * (2 * j + 1) * i) / (2 * N))
    return D


def dct_2d_matrix(block, D):
    temp = np.dot(D, block)
    return np.dot(temp, D.T)

def idct_2d_matrix(dct_block, D):
    temp = np.dot(D.T, dct_block)
    return np.dot(temp, D)

def pad_image_to_block_size(image, block_size):
    h, w = image.shape
    pad_h = (block_size - h % block_size) % block_size
    pad_w = (block_size - w % block_size) % block_size
    if pad_h != 0 or pad_w != 0:
        padded_image = np.pad(image, ((0, pad_h), (0, pad_w)), mode='constant')
        return padded_image, (pad_h, pad_w)
    return image, (0, 0)

def unpad_image(image, padding):
    pad_h, pad_w = padding
    if pad_h != 0 or pad_w != 0:
        return image[:-pad_h, :-pad_w] if pad_h !=0 and pad_w !=0 else \
               image[:-pad_h, :] if pad_h !=0 else \
               image[:, :-pad_w]
    return image

def downsample(Y, Cb, Cr, mode):    
    if mode == "4:2:2":
        # Downsample Cb/Cr horizontally (average 2x1 blocks)
        Cb = (Cb[:, ::2] + Cb[:, 1::2]) / 2
        Cr = (Cr[:, ::2] + Cr[:, 1::2]) / 2
    elif mode == "4:2:0":
        # Downsample Cb/Cr both axes (average 2x2 blocks)
        Cb = (Cb[::2, ::2] + Cb[1::2, ::2] + Cb[::2, 1::2] + Cb[1::2, 1::2]) / 4
        Cr = (Cr[::2, ::2] + Cr[1::2, ::2] + Cr[::2, 1::2] + Cr[1::2, 1::2]) / 4
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    
    return Y, Cb, Cr

def upsample( Cb, Cr, mode):
        """Upsample Cb and Cr components back to original size"""
        if mode == "2":  # 4:2:2
            Cb = np.repeat(Cb, 2, axis=1)
            Cr = np.repeat(Cr, 2, axis=1)
        elif mode == "3":  # 4:2:0
            Cb = np.repeat(np.repeat(Cb, 2, axis=0), 2, axis=1)
            Cr = np.repeat(np.repeat(Cr, 2, axis=0), 2, axis=1)
        return Cb, Cr
    
def get_quantization_tables(quality):
    """Generate quantization tables based on quality (0-50)"""
    # Standard quantization tables (baseline)
    Q_Y_base = np.array([
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99]
    ], dtype=np.float32)

    Q_C_base = np.array([
        [17, 18, 24, 47, 99, 99, 99, 99],
        [18, 21, 26, 66, 99, 99, 99, 99],
        [24, 26, 56, 99, 99, 99, 99, 99],
        [47, 66, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99]
    ], dtype=np.float32)

    # Scale factor calculation (0-50 where 0=max quality, 50=min quality)
    if quality <= 0:
        scale = 1
    elif quality >= 50:
        scale = 50
    else:
        scale = quality
    
    # Scaling formula (non-linear to match JPEG behavior)
    scaling_factor = 1 + (scale / 10)
    
    Q_Y = np.clip(np.round(Q_Y_base * scaling_factor), 1, 255)
    Q_C = np.clip(np.round(Q_C_base * scaling_factor), 1, 255)
    
    return Q_Y, Q_C
def quantize_dct(dct_coeffs, quantization_table): # to quantize DCT coefficients
    """Quantize DCT coefficients using the given quantization table"""
    h, w = dct_coeffs.shape
    quantized = np.zeros_like(dct_coeffs)
    
    for i in range(0, h, 8):
        for j in range(0, w, 8):
            block = dct_coeffs[i:i+8, j:j+8]
            quantized[i:i+8, j:j+8] = np.round(block / quantization_table)
    
    return quantized

def make_dimensions_divisible_by_block_size(img, block_size): # to ensure image dimensions are divisible by block size
    """Adjust image dimensions to be divisible by block_size"""
    h, w = img.shape[:2]
    new_h = (h // block_size) * block_size
    new_w = (w // block_size) * block_size
    return img[:new_h, :new_w]

def log_scale_dct(dct_coeffs): # to apply log scaling for visualization
    # Take absolute value (DCT can have negative values)
    abs_dct = np.abs(dct_coeffs)
    # Add 1 to avoid log(0)
    log_dct = np.log(1 + abs_dct)
    # Normalize to 0-255 for display
    return (log_dct / np.max(log_dct) * 255).astype(np.uint8)
def dequantize_dct(quantized_dct, quantization_table):
    """Reverse the quantization process blockwise"""
    h, w = quantized_dct.shape
    block_size = quantization_table.shape[0]  # Usually 8x8
    
    # Create output array of the same size
    dequantized = np.zeros_like(quantized_dct, dtype=float)
    
    # Process each block
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            # Get current block
            block = quantized_dct[i:i+block_size, j:j+block_size]
            # Apply dequantization (element-wise multiplication with quantization table)
            dequantized_block = block * quantization_table
            # Store back
            dequantized[i:i+block_size, j:j+block_size] = dequantized_block
            
    return dequantized



class JPEGCompressionUI:
    def __init__(self, root):
        self.root = root
        self.root.title("JPEG Compression Tool")
        self.root.geometry("1200x800")
        
        # Set appearance mode and default color theme
        ctk.set_appearance_mode("System")
        ctk.set_default_color_theme("blue")
        
        # Initialize variables
        self.image_path = None
        self.original_image = None
        self.block_size = 8
        self.quality = 10
        
        # Create main frame
        self.main_frame = ctk.CTkFrame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create top panel for controls
        self.control_frame = ctk.CTkFrame(self.main_frame)
        self.control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Create browse button
        self.browse_button = ctk.CTkButton(
            self.control_frame, 
            text="Browse Image", 
            command=self.browse_image
        )
        self.browse_button.pack(side=tk.LEFT, padx=5, pady=5)
        
        # Create filepath display
        self.filepath_var = tk.StringVar(value="No image selected")
        self.filepath_label = ctk.CTkLabel(
            self.control_frame, 
            textvariable=self.filepath_var,
            width=400
        )
        self.filepath_label.pack(side=tk.LEFT, padx=5, pady=5)
        
        # Quality slider
        self.quality_label = ctk.CTkLabel(self.control_frame, text="Quality (0=best, 50=worst):")
        self.quality_label.pack(side=tk.LEFT, padx=(20, 5), pady=5)
        
        self.quality_slider = ctk.CTkSlider(
            self.control_frame, 
            from_=0, 
            to=50, 
            number_of_steps=50,
            command=self.update_quality
        )
        self.quality_slider.set(10)
        self.quality_slider.pack(side=tk.LEFT, padx=5, pady=5)
        
        self.quality_value = ctk.CTkLabel(self.control_frame, text="10")
        self.quality_value.pack(side=tk.LEFT, padx=5, pady=5)
        
        # Process button
        self.process_button = ctk.CTkButton(
            self.control_frame, 
            text="Process Image", 
            command=self.process_image
        )
        self.process_button.pack(side=tk.LEFT, padx=20, pady=5)
        
        # Create display frame for images and results
        self.display_frame = ctk.CTkFrame(self.main_frame)
        self.display_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create notebook/tabview for different steps
        self.tabs = ctk.CTkTabview(self.display_frame)
        self.tabs.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Add tabs
        self.original_tab = self.tabs.add("Original")
        self.ycbcr_tab = self.tabs.add("YCbCr")
        self.dct_tab = self.tabs.add("DCT")
        self.quantized_tab = self.tabs.add("Quantized")
        self.reconstructed_tab = self.tabs.add("Reconstructed")
        self.comparison_tab = self.tabs.add("Comparison")
        
        # Status bar
        self.status_frame = ctk.CTkFrame(self.main_frame, height=30)
        self.status_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.status_var = tk.StringVar(value="Ready")
        self.status_label = ctk.CTkLabel(self.status_frame, textvariable=self.status_var)
        self.status_label.pack(side=tk.LEFT, padx=5, pady=5)
        
        self.psnr_var = tk.StringVar(value="PSNR: N/A")
        self.psnr_label = ctk.CTkLabel(self.status_frame, textvariable=self.psnr_var)
        self.psnr_label.pack(side=tk.RIGHT, padx=5, pady=5)
        
        # Initialize plots frames
        self.setup_tab_contents()
    
    def setup_tab_contents(self):
        """Setup the content areas for each tab"""
        # Original tab
        self.original_frame = ctk.CTkFrame(self.original_tab)
        self.original_frame.pack(fill=tk.BOTH, expand=True)
        
        # YCbCr tab
        self.ycbcr_frame = ctk.CTkFrame(self.ycbcr_tab)
        self.ycbcr_frame.pack(fill=tk.BOTH, expand=True)
        
        # DCT tab
        self.dct_frame = ctk.CTkFrame(self.dct_tab)
        self.dct_frame.pack(fill=tk.BOTH, expand=True)
        
        # Quantized tab
        self.quantized_frame = ctk.CTkFrame(self.quantized_tab)
        self.quantized_frame.pack(fill=tk.BOTH, expand=True)
        
        # Reconstructed tab
        self.reconstructed_frame = ctk.CTkFrame(self.reconstructed_tab)
        self.reconstructed_frame.pack(fill=tk.BOTH, expand=True)
        
        # Comparison tab
        self.comparison_frame = ctk.CTkFrame(self.comparison_tab)
        self.comparison_frame.pack(fill=tk.BOTH, expand=True)
    
    def update_quality(self, value):
        """Update quality value when slider is moved"""
        self.quality = int(value)
        self.quality_value.configure(text=str(self.quality))
    
    def browse_image(self):
        """Open file dialog to select an image"""
        filetypes = [
            ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"),
            ("All files", "*.*")
        ]
        self.image_path = filedialog.askopenfilename(filetypes=filetypes)
        
        if self.image_path:
            self.filepath_var.set(os.path.basename(self.image_path))
            self.original_image = Image.open(self.image_path).convert("RGB")
            self.display_original_image()
    
    def display_original_image(self):
        """Display the original image in the Original tab"""
        for widget in self.original_frame.winfo_children():
            widget.destroy()
        
        # Create figure for matplotlib
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111)
        ax.imshow(self.original_image)
        ax.set_title("Original Image")
        ax.axis("off")
        
        # Embed the matplotlib figure in the tkinter frame
        canvas = FigureCanvasTkAgg(fig, master=self.original_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.status_var.set(f"Loaded image: {os.path.basename(self.image_path)}")

    def process_image(self):
        """Process the image through the JPEG compression pipeline"""
        if self.image_path is None:
            self.status_var.set("Please select an image first")
            return
        
        self.status_var.set("Processing image...")
        self.root.update()
        
        # Convert PIL Image to numpy array
        image = np.array(self.original_image)
        
        # Ensure image dimensions are divisible by block size
        image = make_dimensions_divisible_by_block_size(image, self.block_size)
        
        # Convert RGB to YCrCb
        Y, Cb, Cr = rgb_to_ycbcr(image)
        self.status_var.set("Converted to YCrCb")
        
        # Downsample chrominance channels
        Y_downsampled, Cb_downsampled, Cr_downsampled = downsample(Y, Cb, Cr, "4:2:0")
        self.status_var.set("Downsampled YCrCb")
        
        # Display YCrCb components
        self.show_ycrcb_components(image, Y_downsampled, Cb_downsampled, Cr_downsampled)
        
        # Pad images to be divisible by block size
        Y_padded, Y_padding = pad_image_to_block_size(Y_downsampled, self.block_size)
        Cb_padded, Cb_padding = pad_image_to_block_size(Cb_downsampled, self.block_size)
        Cr_padded, Cr_padding = pad_image_to_block_size(Cr_downsampled, self.block_size)
        
        # DCT transformation
        D = dct_matrix(self.block_size)
        Y_dct = blockwise_transform(Y_padded, self.block_size, dct_2d_matrix, D)
        Cb_dct = blockwise_transform(Cb_padded, self.block_size, dct_2d_matrix, D)
        Cr_dct = blockwise_transform(Cr_padded, self.block_size, dct_2d_matrix, D)
        
        # Display DCT coefficients
        Y_dct_log = log_scale_dct(Y_dct)
        Cb_dct_log = log_scale_dct(Cb_dct)
        Cr_dct_log = log_scale_dct(Cr_dct)
        self.status_var.set("DCT transformation completed")
        self.show_dct(image, Y_dct_log, Cb_dct_log, Cr_dct_log)
        
        # Get quantization tables based on quality
        Q_Y, Q_C = get_quantization_tables(self.quality)
        
        # Quantize the DCT coefficients
        Y_dct_q = quantize_dct(Y_dct, Q_Y)
        Cb_dct_q = quantize_dct(Cb_dct, Q_C)
        Cr_dct_q = quantize_dct(Cr_dct, Q_C)
        
        # Display quantized DCT coefficients
        self.show_quantized(Y_dct_q, Cb_dct_q, Cr_dct_q)
        
        # Dequantize (reverse the quantization)
        Y_dct_dq = dequantize_dct(Y_dct_q, Q_Y)
        Cb_dct_dq = dequantize_dct(Cb_dct_q, Q_C)
        Cr_dct_dq = dequantize_dct(Cr_dct_q, Q_C)
        
        # Inverse DCT
        Y_idct = blockwise_transform(Y_dct_dq, self.block_size, idct_2d_matrix, D)
        Cb_idct = blockwise_transform(Cb_dct_dq, self.block_size, idct_2d_matrix, D)
        Cr_idct = blockwise_transform(Cr_dct_dq, self.block_size, idct_2d_matrix, D)
        
        # Remove padding
        Y_unpadded = unpad_image(Y_idct, Y_padding)
        Cb_unpadded = unpad_image(Cb_idct, Cb_padding)
        Cr_unpadded = unpad_image(Cr_idct, Cr_padding)
        
        # Clip values to valid range
        Y_unpadded = np.clip(Y_unpadded, 0, 255)
        Cb_unpadded = np.clip(Cb_unpadded, 0, 255)
        Cr_unpadded = np.clip(Cr_unpadded, 0, 255)
        
        # Show the IDCT components in the reconstructed tab
        self.show_idct(Y_unpadded, Cb_unpadded, Cr_unpadded)
        
        # Convert back to RGB
        reconstructed_image = ycbcr_to_rgb(Y_unpadded, Cb_unpadded, Cr_unpadded)
        
        # Show the reconstructed image
        self.show_reconstructed(reconstructed_image)
        
        # Show comparison
        self.show_comparison(image, reconstructed_image)
        
        # Calculate PSNR between original and reconstructed images
        psnr_value = psnr(image, reconstructed_image)
        self.psnr_var.set(f"PSNR: {psnr_value:.2f} dB")
        
        self.status_var.set("Processing complete")
    
    def show_ycrcb_components(self, original, Y, Cb, Cr):
        """Display YCbCr components in the YCbCr tab"""
        for widget in self.ycbcr_frame.winfo_children():
            widget.destroy()
        
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        
        # Original image
        axes[0, 0].imshow(original)
        axes[0, 0].set_title("Original Image")
        axes[0, 0].axis("off")
        
        # Y component
        axes[0, 1].imshow(Y, cmap='gray')
        axes[0, 1].set_title("Y (Luminance)")
        axes[0, 1].axis("off")
        
        # Cb component
        axes[1, 0].imshow(Cb, cmap='gray')
        axes[1, 0].set_title("Cb (Blue-difference)")
        axes[1, 0].axis("off")
        
        # Cr component
        axes[1, 1].imshow(Cr, cmap='gray')
        axes[1, 1].set_title("Cr (Red-difference)")
        axes[1, 1].axis("off")
        
        plt.tight_layout()
        
        # Embed the matplotlib figure in the tkinter frame
        canvas = FigureCanvasTkAgg(fig, master=self.ycbcr_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def show_dct(self, original, Y_dct, Cb_dct, Cr_dct):
        """Display DCT coefficients in the DCT tab"""
        for widget in self.dct_frame.winfo_children():
            widget.destroy()
        
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        
        # Original image
        axes[0, 0].imshow(original)
        axes[0, 0].set_title("Original Image")
        axes[0, 0].axis("off")
        
        # Y DCT
        im1 = axes[0, 1].imshow(Y_dct, cmap='gray')
        axes[0, 1].set_title("Y DCT Coefficients")
        axes[0, 1].axis("off")
        
        # Cb DCT
        im2 = axes[1, 0].imshow(Cb_dct, cmap='gray')
        axes[1, 0].set_title("Cb DCT Coefficients")
        axes[1, 0].axis("off")
        
        # Cr DCT
        im3 = axes[1, 1].imshow(Cr_dct, cmap='gray')
        axes[1, 1].set_title("Cr DCT Coefficients")
        axes[1, 1].axis("off")
        
        plt.tight_layout()
        
        # Embed the matplotlib figure in the tkinter frame
        canvas = FigureCanvasTkAgg(fig, master=self.dct_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def show_quantized(self, Y_dct_q, Cb_dct_q, Cr_dct_q):
        """Display quantized DCT coefficients"""
        for widget in self.quantized_frame.winfo_children():
            widget.destroy()
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Y quantized DCT
        axes[0].imshow(np.log(1 + np.abs(Y_dct_q)), cmap='gray')
        axes[0].set_title("Y Quantized DCT")
        axes[0].axis("off")
        
        # Cb quantized DCT
        axes[1].imshow(np.log(1 + np.abs(Cb_dct_q)), cmap='gray')
        axes[1].set_title("Cb Quantized DCT")
        axes[1].axis("off")
        
        # Cr quantized DCT
        axes[2].imshow(np.log(1 + np.abs(Cr_dct_q)), cmap='gray')
        axes[2].set_title("Cr Quantized DCT")
        axes[2].axis("off")
        
        plt.tight_layout()
        
        # Embed the matplotlib figure in the tkinter frame
        canvas = FigureCanvasTkAgg(fig, master=self.quantized_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def show_idct(self, Y, Cb, Cr):
        """Display IDCT components"""
        for widget in self.reconstructed_frame.winfo_children():
            widget.destroy()
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Y component
        axes[0].imshow(Y, cmap='gray')
        axes[0].set_title("Y After IDCT")
        axes[0].axis("off")
        
        # Cb component
        axes[1].imshow(Cb, cmap='gray')
        axes[1].set_title("Cb After IDCT")
        axes[1].axis("off")
        
        # Cr component
        axes[2].imshow(Cr, cmap='gray')
        axes[2].set_title("Cr After IDCT")
        axes[2].axis("off")
        
        plt.tight_layout()
        
        # Embed the matplotlib figure in the tkinter frame
        canvas = FigureCanvasTkAgg(fig, master=self.reconstructed_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def show_reconstructed(self, reconstructed):
        """Display the reconstructed image (will be shown in comparison tab)"""
        pass
    
    def show_comparison(self, original, reconstructed):
        """Display a comparison between original and reconstructed images"""
        for widget in self.comparison_frame.winfo_children():
            widget.destroy()
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        # Original image
        axes[0].imshow(original)
        axes[0].set_title("Original Image")
        axes[0].axis("off")
        
        # Reconstructed image
        axes[1].imshow(reconstructed)
        axes[1].set_title(f"Reconstructed Image (quantization step: {self.quality}, quality: {100 - (self.quality * 2)}%)")
        axes[1].axis("off")
        
        plt.tight_layout()
        
        # Embed the matplotlib figure in the tkinter frame
        canvas = FigureCanvasTkAgg(fig, master=self.comparison_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

# Example of how to use the UI
if __name__ == "__main__":
    root = ctk.CTk()
    app = JPEGCompressionUI(root)
    root.mainloop()