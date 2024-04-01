import cv2
import numpy as np
import matplotlib.pyplot as plt

def driver(image_path):
    image = cv2.imread(image_path) 
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  
    sobelx = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=5)
    sobel_combined = np.sqrt(sobelx**2 + sobely**2)  

    canny_edges = cv2.Canny(gray_image, 100, 200)
    

    plt.figure(figsize=(24, 12))
    
    plt.subplot(1, 4, 1)
    plt.imshow(image_rgb)
    plt.title('Original Image')
    
    plt.subplot(1, 4, 2)
    plt.imshow(gray_image, cmap='gray')
    plt.title('Grayscale Image')
    
    plt.subplot(1, 4, 3)
    plt.imshow(sobel_combined, cmap='gray')
    plt.title('Sobel Edge Detection')
    
    plt.subplot(1, 4, 4)
    plt.imshow(canny_edges, cmap='gray')
    plt.title('Canny Edge Detection')
    
    plt.show()
