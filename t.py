import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.transform import resize  # For better quality upsampling

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


    # Make sure input data is in float format
    Y = Y.astype(np.float32)
    Cb = Cb.astype(np.float32)
    Cr = Cr.astype(np.float32)

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
    B = inv_transform[2, 0] * Y + inv_transform[2, 1] * Cr
    rgb_image = np.clip(np.stack((R, G, B), axis=-1), 0, 255).astype(np.uint8)
    return rgb_image

    
def psnr(original, compressed):
        """Calculate Peak Signal-to-Noise Ratio"""
        mse = np.mean((original.astype(np.float32) - compressed.astype(np.float32)) ** 2)
        return 10 * np.log10(255 ** 2 / mse) if mse != 0 else float('inf')

def show_images(original, transformed, title):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(original)
    axes[0].set_title("Image Originale (RGB)")
    axes[0].axis("off")

    axes[1].imshow(transformed, cmap='gray' if len(transformed.shape) == 2 else None)
    axes[1].set_title(title)
    axes[1].axis("off")

    plt.show()

def show_ycrcb_components(original, Y, Cb, Cr):
    fig, axes = plt.subplots(1, 4, figsize=(16, 5))

    axes[0].imshow(original)
    axes[0].set_title("Original (RGB)")
    axes[0].axis("off")

    axes[1].imshow(Y, cmap='gray')
    axes[1].set_title("Y (Luminance)")
    axes[1].axis("off")

    axes[2].imshow(Cb, cmap='gray')
    axes[2].set_title("Cb (Blue Diff)")
    axes[2].axis("off")

    axes[3].imshow(Cr, cmap='gray')
    axes[3].set_title("Cr (Red Diff)")
    axes[3].axis("off")

    plt.tight_layout()
    plt.show()
    
    
def showDCT(original, Y , Cb , Cr):
    fig, axes = plt.subplots(1, 4, figsize=(16, 5))

    axes[0].imshow(original)
    axes[0].set_title("Original (RGB)")
    axes[0].axis("off")

    axes[1].imshow(Y, cmap='gray')
    axes[1].set_title("DCT Y")
    axes[1].axis("off")

    axes[2].imshow(Cb, cmap='gray')
    axes[2].set_title("DCT Cb")
    axes[2].axis("off")

    axes[3].imshow(Cr, cmap='gray')
    axes[3].set_title("DCT Cr")
    axes[3].axis("off")

    plt.tight_layout()
    plt.show()
    
def blockwise_transform(img, block_size, transform_func, D=None):
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
    # For Y, we want 0-255
    # For Cb/Cr, we want 0-255 (centered at 128)
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
    
    return Y, Cb.astype(np.uint8), Cr.astype(np.uint8)

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
def quantize_dct(dct_coeffs, quantization_table):
    """Quantize DCT coefficients using the given quantization table"""
    h, w = dct_coeffs.shape
    quantized = np.zeros_like(dct_coeffs)
    
    for i in range(0, h, 8):
        for j in range(0, w, 8):
            block = dct_coeffs[i:i+8, j:j+8]
            quantized[i:i+8, j:j+8] = np.round(block / quantization_table)
    
    return quantized

def make_dimensions_divisible_by_block_size(img, block_size):
    """Adjust image dimensions to be divisible by block_size"""
    h, w = img.shape[:2]
    new_h = (h // block_size) * block_size
    new_w = (w // block_size) * block_size
    return img[:new_h, :new_w]

def main():
    image_path = 'ford-mustang-bullitt-1920x774px.jpg'
    image = Image.open(image_path).convert("RGB")
    block_size = 8
    image = make_dimensions_divisible_by_block_size(np.array(image), block_size)
    
    # Convert RGB to YCrCb
    Y, Cb, Cr = rgb_to_ycbcr(image)
    print("Converted to YCrCb")
    print("Cb range:", Cb.min(), Cb.max())
    print("Cr range:", Cr.min(), Cr.max())
    
    Y_downsampled, Cb_downsampled, Cr_downsampled = downsample(Y, Cb, Cr, "4:2:0")
    print("Downsampled YCrCb")
    
    # Show original and YCrCb components
    show_ycrcb_components(image, Y_downsampled, Cb_downsampled, Cr_downsampled)
    
    # Pad images to be divisible by block size
    
    
    Y_padded, Y_padding = pad_image_to_block_size(Y_downsampled, block_size)
    Cb_padded, Cb_padding = pad_image_to_block_size(Cb_downsampled, block_size)
    Cr_padded, Cr_padding = pad_image_to_block_size(Cr_downsampled, block_size)
    
    # DCT transformation
    D = dct_matrix(block_size)
    Y_dct = blockwise_transform(Y_padded, block_size, dct_2d_matrix, D)
    Cb_dct = blockwise_transform(Cb_padded, block_size, dct_2d_matrix, D)
    Cr_dct = blockwise_transform(Cr_padded, block_size, dct_2d_matrix, D)
    
    quality = int(input("Enter quality factor (0-50, where 0=best, 50=worst): "))
    quality = max(0, min(50, quality))
    Q_Y, Q_C = get_quantization_tables(quality)
    # Quantize the DCT coefficients
    dct_y_q = quantize_dct(Y_dct, Q_Y)
    dct_cb_q = quantize_dct(Cb_dct, Q_C)
    dct_cr_q = quantize_dct(Cr_dct, Q_C)
    
    # Unpad the images
    Y_dct = unpad_image(Y_dct, Y_padding)
    Cb_dct = unpad_image(Cb_dct, Cb_padding)
    Cr_dct = unpad_image(Cr_dct, Cr_padding)
    def log_scale_dct(dct_coeffs):
        # Take absolute value (DCT can have negative values)
        abs_dct = np.abs(dct_coeffs)
        # Add 1 to avoid log(0)
        log_dct = np.log(1 + abs_dct)
        # Normalize to 0-255 for display
        return (log_dct / np.max(log_dct) * 255).astype(np.uint8)

    Y_dct_log = log_scale_dct(Y_dct)
    Cb_dct_log = log_scale_dct(Cb_dct)
    Cr_dct_log = log_scale_dct(Cr_dct)
    
    
    print("DCT transformation completed")
    showDCT(image, Y_dct_log , Cb_dct_log, Cr_dct_log)
    
    
    def show_quantized(dct_q_y, dct_q_cb,dct_q_cr):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(np.log(1 + np.abs(dct_q_y)), cmap='gray')
        axes[0].set_title("Quantized Y DCT Coefficients")
        axes[0].axis("off")

        axes[1].imshow(np.log(1 + np.abs(dct_q_cb)), cmap='gray')
        axes[1].set_title("Quantized Cb DCT Coefficients")
        axes[1].axis("off")

        axes[2].imshow(np.log(1 + np.abs(dct_q_cr)), cmap='gray')
        axes[2].set_title("Quantized Cr DCT Coefficients")
        axes[2].axis("off")

        plt.tight_layout()
        plt.show()

    show_quantized(dct_y_q , dct_cb_q, dct_cr_q)
    
    idct_y = blockwise_transform(dct_y_q, block_size, idct_2d_matrix, D)
    idct_cb = blockwise_transform(dct_cb_q, block_size, idct_2d_matrix, D)
    idct_cr = blockwise_transform(dct_cr_q, block_size, idct_2d_matrix, D)
    
    # Get the original ranges from your non-quantized data
    Y_min, Y_max = 0, 255  # Standard Y range
    Cb_min, Cb_max = 0, 255  # Standard Cb range 
    Cr_min, Cr_max = 0, 255  # Standard Cr range

    # Rescale IDCT outputs to match original ranges
    idct_y_rescaled = rescale_idct_output(idct_y, Y_min, Y_max)
    idct_cb_rescaled = rescale_idct_output(idct_cb, Cb_min, Cb_max)
    idct_cr_rescaled = rescale_idct_output(idct_cr, Cr_min, Cr_max)

    # Show the IDCT results
    def show_idct(Y, Cb , Cr):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(Y, cmap='gray')
        axes[0].set_title("IDCT Y")
        axes[0].axis("off")

        axes[1].imshow(Cb, cmap='gray')
        axes[1].set_title("IDCT Cb")
        axes[1].axis("off")

        axes[2].imshow(Cr, cmap='gray')
        axes[2].set_title("IDCT Cr")
        axes[2].axis("off")

        plt.tight_layout()
        plt.show()

    show_idct(idct_y_rescaled, idct_cb_rescaled, idct_cr_rescaled)

    print("Y range (rescaled):", np.min(idct_y_rescaled), np.max(idct_y_rescaled))
    print("Cb range (rescaled):", np.min(idct_cb_rescaled), np.max(idct_cb_rescaled))
    print("Cr range (rescaled):", np.min(idct_cr_rescaled), np.max(idct_cr_rescaled))

    
    
    # show the IDCT results
    
    
    print("Cb range:", idct_cb_rescaled.min(), idct_cb_rescaled.max())
    print("Cr range:", idct_cr_rescaled.min(), idct_cr_rescaled.max())

    # Convert back to RGB using the rescaled values
    rgb_image = ycbcr_to_rgb(idct_y_rescaled, idct_cb_rescaled, idct_cr_rescaled)

    # Now show the reconstructed image
    plt.imshow(rgb_image)
    plt.title("Reconstructed Image")
    plt.axis("off")
    plt.show()

    # Calculate PSNR using NumPy arrays
    original_np = np.array(image)
    reconstructed_np = rgb_image  # Already a NumPy array

    # Make sure dimensions match
    if original_np.shape != reconstructed_np.shape:
        min_height = min(original_np.shape[0], reconstructed_np.shape[0])
        min_width = min(original_np.shape[1], reconstructed_np.shape[1])
        original_np = original_np[:min_height, :min_width]
        reconstructed_np = reconstructed_np[:min_height, :min_width]

    psnr_value = psnr(original_np, reconstructed_np)
    print(f"PSNR: {psnr_value:.2f} dB")

if __name__ == "__main__":
    main()