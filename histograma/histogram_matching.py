import numpy as np
import cv2
import matplotlib.pyplot as plt

def histogram_matching(source, reference):
    # Calcular histograma acumulativo de la imagen fuente
    hist_src, bins_src = np.histogram(source.flatten(), 256, [0,256])
    cdf_src = hist_src.cumsum()
    cdf_src_normalized = cdf_src * float(hist_src.max()) / cdf_src.max()

    # Calcular histograma acumulativo de la imagen de referencia
    hist_ref, bins_ref = np.histogram(reference.flatten(), 256, [0,256])
    cdf_ref = hist_ref.cumsum()
    cdf_ref_normalized = cdf_ref * float(hist_ref.max()) / cdf_ref.max()

    # Mapear valores de la imagen fuente basados en la relación de los CDFs
    cdf_match = np.interp(cdf_src_normalized, cdf_ref_normalized, np.arange(256))

    # Transformar valores de la imagen fuente usando el mapeo obtenido
    result = np.interp(source.flatten(), bins_src[:-1], cdf_match)
    return result.reshape(source.shape)

# Leer imágenes
source_img = cv2.imread('source.jpg', cv2.IMREAD_GRAYSCALE)
reference_img = cv2.imread('reference.jpg', cv2.IMREAD_GRAYSCALE)

# Realizar el histogram matching
matched_img = histogram_matching(source_img, reference_img)

# Mostrar imágenes
plt.figure(figsize=(15, 10))

plt.subplot(1, 3, 1)
plt.imshow(source_img, cmap='gray')
plt.title('Imagen Fuente')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(reference_img, cmap='gray')
plt.title('Imagen de Referencia')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(matched_img, cmap='gray')
plt.title('Imagen Modificada')
plt.axis('off')

plt.tight_layout()
plt.show()
