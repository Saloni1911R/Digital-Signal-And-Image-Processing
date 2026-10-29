import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def clean_binary_image(image_path, kernel_size=5):
    """
    Applies Morphological Opening followed by Closing to remove bright noise 
    and fill dark cracks while preserving object shape.
    """
    # Read binary image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found at: {image_path}")
    
    # Define circular/disk structuring element
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    # 1. Opening: Remove bright noise
    opened = cv2.morphologyEx(img, cv2.MORPH_OPEN, se)
    
    # 2. Closing: Fill dark cracks
    cleaned = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, se)
    
    return img, opened, cleaned

# List of input images
image_files = ["ope61.png", "ope62.png", "ope63.png"]

output_folder = "morphology_outputs"
os.makedirs(output_folder, exist_ok=True)

for img_file in image_files:
    if not os.path.exists(img_file):
        print(f"File skipped (not found): {img_file}")
        continue
    
    base_name = os.path.splitext(img_file)[0]
    
    # Process the image
    original, opening_step, final_result = clean_binary_image(img_file, kernel_size=5)
    
    # Save outputs as separate image files
    cv2.imwrite(os.path.join(output_folder, f"{base_name}_1_original.png"), original)
    cv2.imwrite(os.path.join(output_folder, f"{base_name}_2_after_opening.png"), opening_step)
    cv2.imwrite(os.path.join(output_folder, f"{base_name}_3_final_cleaned.png"), final_result)
    
    # Display each stage in a separate window
    stages = {
        "1_Original_Input": original,
        "2_After_Opening_Noise_Removed": opening_step,
        "3_Final_Result_Cracks_Filled": final_result
    }
    
    for stage_name, stage_img in stages.items():
        plt.figure(figsize=(6, 6))
        plt.imshow(stage_img, cmap='gray')
        plt.title(f"{base_name} - {stage_name}")
        plt.axis('off')
        plt.show()

print(f"Processing complete! All separate outputs are saved in folder: '{output_folder}'")