from PIL import Image
import cv2 
from skimage import io 
import matplotlib.pyplot as plt

#img path
image_path = "/home/saloni/Downloads/chatgpt.png"
pil_imge = Image.open(image_path)

plt.figure(figsize = (6,6))
plt.imshow(pil_imge)
plt.tital("Image Read using Pillow (PIL)")
plt.axis("off")
plt.show()

