# Ejecutar por ejemplo como: 
#     $python3 p1_imagen.py --image_path Mari.jpeg --efecto cojin --save

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
    kernel /= np.sum(kernel) # Normalizar el kernel
    return kernel

def convolution(image, kernel):
    """
    Aplica una convolución bidimensional entre la imagen y el kernel dado.
   
    Args:
        image: La imagen de entrada (numpy array).
        kernel: El kernel a utilizar en la convolución.
     
    Returns:
        La imagen resultante después de la convolución.
    """
    kernel_height, kernel_width = kernel.shape
    image_height, image_width = image.shape
   
    # Padding de la imagen para manejar los bordes
    padded_image = np.pad(image, ((kernel_height//2, kernel_height//2), (kernel_width//2, kernel_width//2)), mode='constant')
   
    # Crear la imagen resultante
    result_image = np.zeros_like(image)
   
    # Realizar la convolución
    for y in range(image_height):
        for x in range(image_width):
            result_image[y, x] = np.sum(padded_image[y:y+kernel_height, x:x+kernel_width] * kernel)
   
    return result_image


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

def aplicar_efectos(image, filtro):
    if filtro == 'contraste':
        # HACER NOSOTROS
        # Efecto de mejora del contraste
        alpha = 1.5  # Factor de aumento de contraste
        beta = 30    # Desplazamiento
        frame = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

        frame_gris = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = equalizeHist(frame_gris)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif filtro == 'alien':
        # Filtro Alien: Cambiar el color de la piel a verde
        frame_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(frame_hsv, lower_skin, upper_skin)
        image[mask > 0] = [0, 255, 0]  # Cambiar a color verde
    elif filtro == 'poster':
        # Filtro Póster: Reducir el número de colores
        num_colores = 8
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = (image // (256 // num_colores)) * (256 // num_colores)
    elif filtro == 'barril' or filtro == 'cojin':
        k1, k2 = 0.1, 0.1  # Coeficientes de distorsión
        
        if filtro == 'cojin':
            k1, k2 = -k1, -k2

        alto, ancho = image.shape[:2]

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
        image = cv2.remap(image, x_distorsion, y_distorsion, cv2.INTER_LINEAR)
    elif filtro == 'blur':
        # Filtro de desenfoque
        image = gaussian_blur(image, 15, 1.5)
    return image

def main(image_path, filtro):
    # Leer la imagen desde la ruta proporcionada
    image = cv2.imread(image_path)

    if image is None:
        print("No se pudo leer la imagen.")
        return

    # Aplicar efectos
    image_efecto = aplicar_efectos(image, filtro)

    # Mostrar el resultado en pantalla
    cv2.imshow('Filtro ' + filtro.capitalize(), image_efecto)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return image_efecto

if __name__ == "__main__":
    # Argumentos de línea de comandos
    parser = argparse.ArgumentParser(description='Aplicar efectos a una imagen.')
    parser.add_argument('--image_path', type=str, help='Ruta de la imagen a procesar.')
    parser.add_argument('--efecto', type=str, default='contraste', choices=['contraste', 'ecualizacion', 'alien', 'poster', 'barril', 'cojin', 'blur'],
                        help='Elegir el filtro a aplicar (contraste, ecualizacion, alien, poster, barril, cojin, blur)')
    parser.add_argument('--save', action='store_true', help='Guardar la imagen resultante')
    args = parser.parse_args()

    output_image_path = None  # Inicializa la variable para guardar la ruta de la imagen de salida
    if args.save:
        # Construir el nombre de archivo para guardar la imagen resultante
        filename_parts = args.image_path.split('.')
        output_image_path = f"{filename_parts[0]}_{args.efecto}.{filename_parts[-1]}"


    # Llamar a la función main con los argumentos proporcionados

    processed_image = main(args.image_path, args.efecto)  # Obtener la imagen procesada desde la función main

    if args.save and output_image_path:
        # Guardar la imagen procesada si la opción --save está activada y se proporcionó una ruta de salida
        cv2.imwrite(output_image_path, processed_image)
        print(f"Imagen guardada como {output_image_path}")