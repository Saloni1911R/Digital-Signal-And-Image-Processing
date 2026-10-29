import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def calculate_metrics(original, processed):
    """Calculates PSNR and SSIM between original and processed images."""
    psnr = cv2.PSNR(original, processed)
    
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2
    
    img1 = original.astype(np.float64)
    img2 = processed.astype(np.float64)
    
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.T)
    
    mu1 = cv2.filter2D(img1, -1, window)
    mu2 = cv2.filter2D(img2, -1, window)
    
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = cv2.filter2D(img1**2, -1, window) - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window) - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window) - mu1_mu2
    
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return psnr, ssim_map.mean()

def alpha_trimmed_mean_filter(image, kernel_size=3, alpha=2):
    """Applies Alpha-Trimmed Mean Filter to remove extreme noise values."""
    pad = kernel_size // 2
    padded_img = np.pad(image, pad, mode='reflect')
    filtered_img = np.zeros_like(image, dtype=np.float64)
    
    rows, cols = image.shape
    for i in range(rows):
        for j in range(cols):
            window = padded_img[i:i + kernel_size, j:j + kernel_size].flatten()
            sorted_window = np.sort(window)
            if alpha > 0 and 2 * alpha < len(sorted_window):
                trimmed_window = sorted_window[alpha:-alpha]
            else:
                trimmed_window = sorted_window
            filtered_img[i, j] = np.mean(trimmed_window)
            
    return np.uint8(np.clip(filtered_img, 0, 255))

# Input dataset images
image_files = ["ope51.png", "ope52.png", "ope53.png", "ope54.png"]
output_folder = "combined_frame_outputs"
os.makedirs(output_folder, exist_ok=True)

# Non-linear filter dictionary
filters = {
    "Original": lambda img: img,
    "Median 3x3": lambda img: cv2.medianBlur(img, 3),
    "Median 5x5": lambda img: cv2.medianBlur(img, 5),
    "Median 7x7": lambda img: cv2.medianBlur(img, 7),
    "Alpha-Trimmed 3x3": lambda img: alpha_trimmed_mean_filter(img, kernel_size=3, alpha=2),
    "Min Filter 3x3": lambda img: cv2.erode(img, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))),
    "Max Filter 3x3": lambda img: cv2.dilate(img, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
}

for img_file in image_files:
    if not os.path.exists(img_file):
        print(f"File skipped (not found): {img_file}")
        continue
        
    img = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
    base_name = os.path.splitext(img_file)[0]
    
    processed_results = []
    table_data = []
    
    # Process all filters for the current image
    for filter_name, filter_func in filters.items():
        res = filter_func(img)
        psnr, ssim = calculate_metrics(img, res)
        
        processed_results.append((filter_name, res))
        
        # Build quantitative comparison table rows
        if filter_name == "Original":
            table_data.append([filter_name, "Baseline", "1.0000"])
        else:
            table_data.append([filter_name, f"{psnr:.2f} dB", f"{ssim:.4f}"])
    
    # Create single grid layout: 2 rows of images + 1 row for metrics table at bottom
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(f"Non-Linear Filtering Analysis: {img_file}", fontsize=16, fontweight='bold')
    
    # Grid specification: 3 rows x 4 columns
    gs = fig.add_gridspec(3, 4, height_ratios=[1, 1, 0.6])
    
    # Plot top 7 images on grid
    for idx, (title, out_img) in enumerate(processed_results):
        r = idx // 4
        c = idx % 4
        ax = fig.add_subplot(gs[r, c])
        ax.imshow(out_img, cmap='gray')
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.axis('off')
    
    # Leave the 8th slot clear if empty
    ax_empty = fig.add_subplot(gs[1, 3])
    ax_empty.axis('off')
    
    # Create Table spanning full width at bottom row
   # Create Table spanning full width at bottom row
    ax_table = fig.add_subplot(gs[2, :])
    ax_table.axis('off')
    
    col_labels = ["Filter Type", "PSNR (Peak Signal-to-Noise Ratio)", "SSIM (Structural Similarity)"]
    table = ax_table.table(
        cellText=table_data,
        colLabels=col_labels,
        loc='center',
        cellLoc='center'
    )
    
    # Fix for AttributeError:
    table.auto_set_font_size(False)
    table.set_fontsize(10)  # Fixed method name
    table.scale(1, 1.4)
    
    # Highlight table headers
    for key, cell in table.get_celld().items():
        if key[0] == 0:
            cell.set_facecolor('#40466e')
            cell.get_text().set_color('white')
            cell.get_text().set_weight('bold')
    
    plt.tight_layout()
    
    # Save the combined frame image with table
    frame_save_path = os.path.join(output_folder, f"{base_name}_frame_with_table.png")
    plt.savefig(frame_save_path, dpi=300, bbox_inches='tight')
    
    # Display the frame with table
    plt.show()

print(f"All combined frames and tables saved inside: '{output_folder}'")