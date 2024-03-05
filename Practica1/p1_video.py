# Ejecutar por ejemplo como: 
#     $python3 p1.py --filtro contraste

import cv2
import numpy as np
import argparse

def gaussian_blur(image, kernel_size, sigma):
    """
    Aplica un filtro Gaussiano a la imagen dada.
    
    Args:
        image: La imagen de entrada (numpy array).
        kernel_size: El tamaño del kernel Gaussiano (debe ser un número impar).
        sigma: El desvío estándar del kernel Gaussiano.
    
    Returns:
        La imagen suavizada.
    """
    # Crear un kernel Gaussiano
    kernel = gaussian_kernel(kernel_size, sigma)
    
    # Aplicar convolución
    blurred_image = convolution(image, kernel)
    
    return blurred_image

def gaussian_kernel(kernel_size, sigma):
    """
    Genera un kernel Gaussiano de tamaño kernel_size x kernel_size y con el desvío estándar dado.
    
    Args:
        kernel_size: El tamaño del kernel (debe ser un número impar).
        sigma: El desvío estándar del kernel Gaussiano.
        
    Returns:
        El kernel Gaussiano.
    """
    kernel = np.fromfunction(lambda x, y: (1/(2*np.pi*sigma**2)) * np.exp(-((x - (kernel_size-1)/2)**2 + (y - (kernel_size-1)/2)**2)/(2*sigma**2)), (kernel_size, kernel_size))
    kernel /= np.sum(kernel)  # Normalizar el kernel
    return kernel

def convolution(image, kernel):
    """
    Aplica una convolución entre la imagen y el kernel dado.
    
    Args:
        image: La imagen de entrada (numpy array).
        kernel: El kernel a utilizar en la convolución.
        
    Returns:
        La imagen resultante después de aplicar la convolución.
    """
    # Extraer dimensiones de la imagen y del kernel
    image_height, image_width, num_channels = image.shape
    kernel_height, kernel_width = kernel.shape
    
    # Calcular la cantidad de píxeles a agregar alrededor de la imagen para poder aplicar el kernel completo
    pad_height = kernel_height // 2
    pad_width = kernel_width // 2
    
    # Crear una imagen con relleno
    padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width), (0, 0)), mode='constant')
    
    # Inicializar la imagen resultante
    result_image = np.zeros_like(image)
    
    # Aplicar la convolución en cada canal por separado
    for c in range(num_channels):
        for y in range(image_height):
            for x in range(image_width):
                # Extraer la región de la imagen que se superpone con el kernel
                region = padded_image[y:y+kernel_height, x:x+kernel_width, c]
                # Realizar la convolución y sumar los resultados
                result_image[y, x, c] = np.sum(region * kernel)
    
    return result_image.astype(np.uint8)


def calcHist(image):
    # Convertir la imagen a escala de grises si es necesario
    if len(image.shape) > 2:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Inicializar el histograma con ceros
    hist = np.zeros(256, dtype=int)

    # Calcular el histograma
    for pixel in np.nditer(gray):
        hist[pixel] += 1

    return hist

def manual_convertScaleAbs(image, alpha, beta):
    # Aplicar la transformación lineal: multiplicación por alpha y suma de beta
    transformed_image = image.astype(float) * alpha + beta
    
    # Asegurar que los valores de píxeles estén dentro del rango correcto (0 a 255)
    transformed_image = np.clip(transformed_image, 0, 255)
    
    # Convertir los valores de píxeles a enteros sin signo de 8 bits
    transformed_image = transformed_image.astype(np.uint8)
    
    return transformed_image

def equalizeHist(image):
    # Convertir la imagen a escala de grises si es necesario
    if len(image.shape) > 2:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Calcular el histograma de la imagen en escala de grises
    # Nota: Hemos utilizado el histograma que ofrece openvc en vez de el nuestro por eficiencia, pero el nuestro funciona igual    
    hist = cv2.calcHist([gray], [0], None, [256], [0,256])

    # Calcular el histograma acumulativo
    cumulative_hist = hist.cumsum()

    # Normalizar el histograma acumulativo
    cumulative_hist_normalized = cumulative_hist / cumulative_hist[-1]

    # Mapear los valores de intensidad de píxeles originales a los nuevos valores de intensidad
    equalized_image = (cumulative_hist_normalized[gray] * 255).astype('uint8')

    return equalized_image

def aplicar_efectos(frame, filtro):
    if filtro == 'contraste':
        # HACER NOSOTROS
        # Efecto de mejora del contraste
        alpha = 1.5  # Factor de aumento de contraste
        beta = 30    # Desplazamiento
        frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)

        frame_gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = equalizeHist(frame_gris)
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    elif filtro == 'alien':
        # Filtro Alien: Cambiar el color de la piel a verde
        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(frame_hsv, lower_skin, upper_skin)
        frame[mask > 0] = [0, 255, 0]  # Cambiar a color verde
    elif filtro == 'poster':
        # Filtro Póster: Reducir el número de colores
        num_colores = 8
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = (frame // (256 // num_colores)) * (256 // num_colores)
    elif filtro == 'barril' or filtro == 'cojin':
        k1, k2 = 0.1, 0.1  # Coeficientes de distorsión
        
        if filtro == 'cojin':
            k1, k2 = -k1, -k2

        alto, ancho = frame.shape[:2]

        # Calcular el centro de la imagen
        centro_y = alto / 2
        centro_x = ancho / 2

        # Generar un array con las coordenadas de los píxeles
        x, y = np.meshgrid(np.arange(ancho), np.arange(alto))

        # Transformar a float
        x = x.astype(np.float32)
        y = y.astype(np.float32)

        # Normalizar
        x_normalizada = (x - centro_x) / centro_x
        y_normalizada = (y - centro_y) / centro_y

        r = x_normalizada**2 + y_normalizada**2

        x_distorsion = x_normalizada * (1 + k1 * r + k2 * r**2)
        y_distorsion = y_normalizada * (1 + k1 * r + k2 * r**2)

        # Deshacer la normalización
        x_distorsion = (x_distorsion * centro_x) + centro_x
        y_distorsion = (y_distorsion * centro_y) + centro_y

        # Aplicar la distorsión a la imagen usando remap
        frame = cv2.remap(frame, x_distorsion, y_distorsion, cv2.INTER_LINEAR)

    elif filtro == 'blur':
        # Filtro de desenfoque
        frame = gaussian_blur(frame, 15, 0.1)
    return frame
def main():
    # Argumentos de línea de comandos
    parser = argparse.ArgumentParser(description='Aplicar efectos a una transmisión de cámara en vivo.')
    parser.add_argument('--filtro', type=str, default='contraste', choices=['contraste', 'ecualizacion', 'alien', 'poster', 'barril', 'cojin', 'blur'],
                        help='Elegir el filtro a aplicar (contraste, ecualizacion, alien, poster, barril, cojin, blur)')
    args = parser.parse_args()

    # Inicializar la cámara
    captura = cv2.VideoCapture(0)

    if not captura.isOpened():
        print("Error al abrir la cámara.")
        exit()

    while True:
        ret, frame = captura.read()

        if not ret:
            print("Error al capturar el frame.")
            break

        # Aplicar efectos
        frame_efecto = aplicar_efectos(frame, args.filtro)

        # Mostrar el resultado en pantalla
        cv2.imshow('Filtro ' + args.filtro.capitalize(), frame_efecto)

        # Salir del bucle si se presiona la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Liberar la cámara y cerrar las ventanas
    captura.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()