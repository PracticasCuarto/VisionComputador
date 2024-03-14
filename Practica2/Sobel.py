import cv2
import numpy as np

# Cargar la imagen
image = cv2.imread('Contornos/poster.pgm', cv2.IMREAD_GRAYSCALE)

# Aplicar filtrado Gaussiano
image_blurred = cv2.GaussianBlur(image, (5, 5), 0)

# Calcular el gradiente horizontal y vertical con Sobel
gradient_x = cv2.Sobel(image_blurred, cv2.CV_64F, 1, 0, ksize=3)
gradient_y = cv2.Sobel(image_blurred, cv2.CV_64F, 0, 1, ksize=3)
gradient_y_inverted = gradient_y * -1

# Normalizar los valores del gradiente al rango [0, 255]
normalized_gradient_x = cv2.normalize(gradient_x, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
normalized_gradient_y = cv2.normalize(gradient_y, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
normalized_gradient_y_inverted = cv2.normalize(gradient_y_inverted, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

# Ajustar los valores para plotear
adjusted_gradient_x = cv2.addWeighted(normalized_gradient_x, 0.5, np.zeros_like(normalized_gradient_x) + 128, 0.5, 0)
adjusted_gradient_y = cv2.addWeighted(normalized_gradient_y, 0.5, np.zeros_like(normalized_gradient_y) + 128, 0.5, 0)
adjusted_gradient_y_inverted = cv2.addWeighted(normalized_gradient_y_inverted, 0.5, np.zeros_like(normalized_gradient_y_inverted) + 128, 0.5, 0)

# Calcular el módulo y la orientación del gradiente
gradient_magnitude, gradient_orientation = cv2.cartToPolar(gradient_x, gradient_y)

# Normalizar los valores del módulo del gradiente al rango [0, 255]
normalized_magnitude = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

# Normalizar los valores de la orientación del gradiente al rango [0, 2*pi]
normalized_orientation = (gradient_orientation + np.pi) % (2 * np.pi)

# Convertir los valores de la orientación del gradiente al rango [0, 255]
normalized_orientation_255 = (normalized_orientation / np.pi * 128).astype(np.uint8)

# Aplicar umbral para resaltar bordes
threshold_value = 20
bordes = cv2.threshold(normalized_magnitude, threshold_value, 255, cv2.THRESH_BINARY)[1]

# Mostrar los resultados
cv2.imshow('Gradiente Horizontal', adjusted_gradient_x)
cv2.imshow('Gradiente Vertical', adjusted_gradient_y_inverted)
cv2.imshow('Modulo del Gradiente', normalized_magnitude)
cv2.imshow('Orientacion del Gradiente', normalized_orientation_255)
cv2.imshow('Bordes', bordes)
cv2.waitKey(0)
cv2.destroyAllWindows()
