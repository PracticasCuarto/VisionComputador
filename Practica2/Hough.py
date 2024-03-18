import math
import numpy as np
import cv2
from collections import Counter
from collections import defaultdict


def grados_a_radianes(grados):
    # Convertir a radianes
    radianes = grados * math.pi / 180
    # Ajustar el resultado para que esté dentro del rango 0 - 2π
    radianes_ajustados = radianes % (2 * math.pi)
    return radianes_ajustados

def calcular_interseccion(punto, direccion, y_linea):
    # Desempaquetamos las coordenadas del punto
    x_punto, y_punto = punto
    
    # Calculamos la pendiente de la línea utilizando la dirección
    pendiente = math.tan(direccion)
    
    # Calculamos la coordenada x del punto de intersección
    x_interseccion = (y_linea - y_punto + pendiente * x_punto) / pendiente
    
    # La coordenada y del punto de intersección será la línea horizontal
    y_interseccion = y_linea

    if (x_interseccion < 0 or x_interseccion > image.shape[1]):
        # La interseccion esta en el infinito
        x_interseccion = None
    
    return x_interseccion, y_interseccion

def encontrar_valor_aproximado_puntos(puntos, margen=2):
    frecuencias = defaultdict(int)

    # Paso 1: Redondear las coordenadas x a múltiplos de 10
    for punto in puntos:
        x, _ = punto
        intervalo = round(x / margen) * margen
        frecuencias[intervalo] += 1

    # Paso 3: Encontrar el intervalo con la frecuencia máxima
    valor_aproximado = max(frecuencias, key=frecuencias.get)

    # Paso 4: Calcular el punto medio del intervalo
    punto_medio = valor_aproximado + margen / 2

    return punto_medio

# Cargar la imagen
image = cv2.imread('Contornos/pasillo2.pgm', cv2.IMREAD_GRAYSCALE)

# Aplicar filtrado Gaussiano
image_blurred = cv2.GaussianBlur(image, (5, 5), 0)

# Calcular el gradiente horizontal y vertical con Sobel
gradient_x = cv2.Sobel(image_blurred, cv2.CV_64F, 1, 0, ksize=3)
gradient_y= cv2.Sobel(image_blurred, cv2.CV_64F, 0, 1, ksize=3)

# Calcular el módulo y la orientación del gradiente
gradient_magnitude, gradient_orientation = cv2.cartToPolar(gradient_x, gradient_y)

# Normalizar los valores de la orientación del gradiente al rango [0, 2*pi]
normalized_orientation = (gradient_orientation + np.pi) % (2 * np.pi)

# Normalizar los valores del módulo del gradiente al rango [0, 255]
normalized_magnitude = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

cv2.imshow('Bordes', normalized_magnitude)

# Umbral para el gradiente
gradient_threshold = 100

# Obtener coordenadas de puntos que superan el umbral de gradiente
y_coords, x_coords = np.where(gradient_magnitude > gradient_threshold)

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
    gradiente = normalized_orientation[y_coords[i], x_coords[i]]
    en_rango1 = (gradiente < umbral_descarte1)
    en_rango2 = (gradiente > umbral_descarte2 and gradiente < umbral_descarte3)
    en_rango3 = (gradiente > umbral_descarte4 and gradiente < umbral_descarte5)
    en_rango4 = (gradiente > umbral_descarte6 and gradiente < umbral_descarte7)
    en_rango5 = (gradiente > umbral_descarte8)
    valido = en_rango1 or en_rango2 or en_rango3 or en_rango4 or en_rango5
    if not valido:
        filtered_x_coords.append(x_coords[i])
        filtered_y_coords.append(y_coords[i])
        filtered_gradient_orientation.append(gradiente + np.pi / 2)  # Rotar 90 grados para que sea perpendicular al borde

# Combina los arrays de coordenadas x e y en un único array bidimensional
puntos = np.column_stack((filtered_x_coords, filtered_y_coords))

# Coordenadas de la línea de horizonte (suponiendo que está en el centro de la imagen)
horizon_line_y = image.shape[0] // 2  # Se toma la mitad de la altura de la imagen

# Calcular las intersecciones con la línea del horizonte
intersecciones = []
for punto, direccion in zip(puntos, filtered_gradient_orientation):
    x, y = punto
    # La ecuación paramétrica de una línea es: x = x0 + t*cos(θ), y = y0 + t*sin(θ)
    # Donde (x0, y0) es el punto inicial y θ es el ángulo de la dirección
    puntoInterseccion = calcular_interseccion(punto, direccion, horizon_line_y)
    if puntoInterseccion[0] != None:
        intersecciones.append(puntoInterseccion)

punto_mas_votado = encontrar_valor_aproximado_puntos(intersecciones)

print("El punto más votado es:", punto_mas_votado)

x = int(punto_mas_votado)  # Convertir a entero
y =  horizon_line_y

cv2.line(image, (x - 5, y), (x + 5, y), (0, 255, 0), 2)  # Línea horizontal
cv2.line(image, (x, y - 5), (x, y + 5), (0, 255, 0), 2)  # Línea vertical

# Mostrar la imagen con los puntos de intersección
cv2.imshow("Intersecciones con la linea del horizonte", image)

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