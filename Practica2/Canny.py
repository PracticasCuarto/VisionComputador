import cv2
import numpy as np

def gaussian_filter(image, sigma):
    """
    Aplica un filtro gaussiano a la imagen para suavizarla.
    """
    return cv2.GaussianBlur(image, (0, 0), sigma)

def compute_gradient(image):
    """
    Calcula el gradiente de la imagen en dirección x e y.
    """
    gradient_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    direction = np.arctan2(gradient_y, gradient_x)
    return magnitude, direction

def non_maximum_suppression(magnitude, direction):
    """
    Suprime los píxeles que no son máximos en la dirección del gradiente.
    """
    suppressed_magnitude = np.zeros_like(magnitude)
    direction = np.rad2deg(direction) % 180
    for i in range(1, magnitude.shape[0] - 1):
        for j in range(1, magnitude.shape[1] - 1):
            angle = direction[i, j]
            if (0 <= angle < 22.5) or (157.5 <= angle <= 180):
                neighbors = [magnitude[i, j-1], magnitude[i, j+1]]
            elif (22.5 <= angle < 67.5):
                neighbors = [magnitude[i-1, j-1], magnitude[i+1, j+1]]
            elif (67.5 <= angle < 112.5):
                neighbors = [magnitude[i-1, j], magnitude[i+1, j]]
            else:
                neighbors = [magnitude[i-1, j+1], magnitude[i+1, j-1]]
            if magnitude[i, j] >= max(neighbors):
                suppressed_magnitude[i, j] = magnitude[i, j]
    return suppressed_magnitude

def double_thresholding(image, low_threshold, high_threshold):
    """
    Realiza umbralización doble para identificar bordes fuertes y débiles.
    """
    strong_edges = (image >= high_threshold)
    weak_edges = (image >= low_threshold) & (image < high_threshold)
    return strong_edges, weak_edges

def edge_tracking(strong_edges, weak_edges):
    """
    Realiza el seguimiento de bordes utilizando la técnica de histéresis.
    """
    edge_image = np.zeros_like(strong_edges)
    edge_image[strong_edges] = 255
    rows, cols = edge_image.shape
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (-1, 1), (1, -1)]
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            if weak_edges[i, j]:
                for direction in directions:
                    neighbor_i, neighbor_j = i + direction[0], j + direction[1]
                    if strong_edges[neighbor_i, neighbor_j]:
                        edge_image[i, j] = 255
                        break
    return edge_image

def canny_edge_detection(image, sigma, low_threshold, high_threshold):
    """
    Implementación del algoritmo de detección de bordes de Canny.
    """
    smoothed_image = gaussian_filter(image, sigma)
    gradient_magnitude, gradient_direction = compute_gradient(smoothed_image)
    suppressed_magnitude = non_maximum_suppression(gradient_magnitude, gradient_direction)
    strong_edges, weak_edges = double_thresholding(suppressed_magnitude, low_threshold, high_threshold)
    edge_image = edge_tracking(strong_edges, weak_edges)
    return edge_image

# Cargar la imagen
image = cv2.imread('Contornos/poster.pgm', cv2.IMREAD_GRAYSCALE)

# Definir los parámetros
sigma = 0.5
low_threshold = 10
high_threshold = 50

# Detectar bordes utilizando Canny
edges = canny_edge_detection(image, sigma, low_threshold, high_threshold)
edges_display1 = np.uint8(edges)
# Mostrar la imagen de bordes
cv2.imshow("Bordes Detectados (Canny)", edges_display1 * 255)
cv2.waitKey(0)
cv2.destroyAllWindows()
