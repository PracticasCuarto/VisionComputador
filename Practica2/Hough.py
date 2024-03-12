import math
import numpy as np
import cv2


def grados_a_radianes(grados):
    # Convertir a radianes
    radianes = grados * math.pi / 180
    # Ajustar el resultado para que esté dentro del rango 0 - 2π
    radianes_ajustados = radianes % (2 * math.pi)
    return radianes_ajustados

# Cargar la imagen
image = cv2.imread('Contornos/pasillo1.pgm', cv2.IMREAD_GRAYSCALE)

# Aplicar filtrado Gaussiano
image_blurred = cv2.GaussianBlur(image, (5, 5), 0)

# Calcular el gradiente horizontal y vertical con Sobel
gradient_x = cv2.Sobel(image_blurred, cv2.CV_64F, 1, 0, ksize=3)
gradient_y= cv2.Sobel(image_blurred, cv2.CV_64F, 0, 1, ksize=3)

# Calcular el módulo y la orientación del gradiente
gradient_magnitude, gradient_orientation = cv2.cartToPolar(gradient_x, gradient_y)

# Normalizar los valores del módulo del gradiente al rango [0, 255]
normalized_magnitude = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

cv2.imshow('Bordes', normalized_magnitude)

# Umbral para el gradiente
gradient_threshold = 100

# Obtener coordenadas de puntos que superan el umbral de gradiente
y_coords, x_coords = np.where(gradient_magnitude > gradient_threshold)

# Mostrar las orientaciones del gradiente
print(gradient_orientation) 

# Mostrar maximo y minimo
print(np.max(gradient_orientation))
print(np.min(gradient_orientation))

# Filtrar los puntos que tienen una orientación cercana a horizontal o vertical
filtered_x_coords = []
filtered_y_coords = []
filtered_gradient_orientation = []

# Declarar el umbral de descartes que es el 15% de 2pi  
umbral_descarte1 = grados_a_radianes(15)
umbral_descarte2 = grados_a_radianes(75)
umbral_descarte3 = grados_a_radianes(105)
umbral_descarte4 = grados_a_radianes(165)
umbral_descarte5 = grados_a_radianes(195)
umbral_descarte6 = grados_a_radianes(255)
umbral_descarte7 = grados_a_radianes(285)
umbral_descarte8 = grados_a_radianes(345)

for i in range(len(x_coords)):
    gradiente = gradient_orientation[y_coords[i], x_coords[i]]
    en_rango1 = (gradiente < umbral_descarte1)
    en_rango2 = (gradiente > umbral_descarte2 and gradiente < umbral_descarte3)
    en_rango3 = (gradiente > umbral_descarte4 and gradiente < umbral_descarte5)
    en_rango4 = (gradiente > umbral_descarte6 and gradiente < umbral_descarte7)
    en_rango5 = (gradiente > umbral_descarte8)
    valido = en_rango1 or en_rango2 or en_rango3 or en_rango4 or en_rango5
    if not valido:
        filtered_x_coords.append(x_coords[i])
        filtered_y_coords.append(y_coords[i])
        filtered_gradient_orientation.append(gradiente)
        

# Filtrar de gradient orientation los valores que no esten en el rango





# Dibujar los puntos que votan en una imagen en blanco
voting_image = np.zeros_like(image)
voting_image[filtered_y_coords, filtered_x_coords] = 255  # Pintar los puntos que votan de blanco (255)

# Coordenadas de la línea de horizonte (suponiendo que está en el centro de la imagen)
horizon_line_y = image.shape[0] // 2  # Se toma la mitad de la altura de la imagen

# Dibujar la línea de horizonte en la imagen de votos
voting_image_with_horizon = cv2.line(voting_image, (0, horizon_line_y), (image.shape[1], horizon_line_y), (255, 255, 255), 1)

# Trazar líneas desde los puntos filtrados hacia la línea de horizonte
for x, y in zip(filtered_x_coords, filtered_y_coords):
    cv2.line(voting_image_with_horizon, (x, y), (x, horizon_line_y), (128, 128, 128), 1)

# Mostrar la imagen con líneas trazadas
cv2.imshow('Puntos que votan y líneas trazadas', voting_image_with_horizon)

cv2.waitKey(0)
cv2.destroyAllWindows()

# Dibujar una cruz en el punto de fuga
# img_with_cross = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
# cv2.drawMarker(img_with_cross, (focal_point, img.shape[0] // 2), (0, 255, 0), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)

# # Mostrar la imagen con la cruz
# cv2.imshow('Image with Cross', img_with_cross)
# cv2.waitKey(0)
# cv2.destroyAllWindows()



# Gradiente en funcion del umbral, pintarlos, y decir que esos votan
# para quitar las lineas con la orientacion del gradiente (15)