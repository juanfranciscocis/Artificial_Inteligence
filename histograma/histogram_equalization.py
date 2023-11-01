import cv2
import numpy as np
import matplotlib.pyplot as plt

# Leer imagen en escala de grises
img = cv2.imread('../histograma/IMG_6117.jpeg', cv2.IMREAD_GRAYSCALE)

# Equalizar histograma
img_equalizada = cv2.equalizeHist(img)

# Mostrar imágenes antes y después de la equalización
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray')
plt.title('Imagen Original')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(img_equalizada, cmap='gray')
plt.title('Imagen Equalizada')
plt.axis('off')

plt.tight_layout()
plt.show()

# Mostrar histogramas antes y después de la equalización
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.hist(img.ravel(),256,[0,256])
plt.title('Histograma Original')

plt.subplot(1, 2, 2)
plt.hist(img_equalizada.ravel(),256,[0,256])
plt.title('Histograma Equalizado')

plt.show()
