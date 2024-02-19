import cv2
import numpy as np
import argparse
import concurrent.futures

def calcHistHilos(image):
    # Convertir la imagen a escala de grises si es necesario
    if len(image.shape) > 2:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Inicializar el histograma con ceros
    hist = np.zeros(256, dtype=int)

    # Función para contar la ocurrencia de píxeles en una subsección de la imagen
    def count_pixels(start, end):
        local_hist = np.zeros(256, dtype=int)
        for pixel in np.nditer(gray[start:end]):
            local_hist[pixel] += 1
        return local_hist

    # Definir el número de hilos a utilizar (puedes ajustar este valor según tu CPU)
    num_threads = 7  # Por ejemplo, usar 4 hilos

    # Dividir la imagen en secciones para que cada hilo procese una parte
    section_size = len(gray) // num_threads
    sections = [(i * section_size, (i + 1) * section_size) for i in range(num_threads - 1)]
    sections.append(((num_threads - 1) * section_size, len(gray)))

    # Utilizar hilos para calcular el histograma de forma concurrente
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(count_pixels, start, end) for start, end in sections]

        # Combinar los resultados de los hilos en un único histograma
        for future in concurrent.futures.as_completed(futures):
            hist += future.result()

    return hist

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
    hist = calcHistHilos(gray)

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
    elif filtro == 'distorsion':
        # Transformacion inversa usar remap o algo asi
        # Hay una implementación en moodle en matlab de una rotacion que se puede 
        # usar como base para hacer la transformación inversa
        # Efecto de distorsión de barril y cojín
        # Los siguientes parámetros son de ejemplo
        k1, k2, k3 = 0.8, 0.0, 0.0  # Coeficientes de distorsión radial
        p1, p2 = 0.0, 0.0  # Coeficientes de distorsión tangencial
        fx, fy = 500, 500  # Distancia focal en píxeles (ejemplo)
        cx, cy = frame.shape[1] // 2, frame.shape[0] // 2  # Punto principal (centro de la imagen)

        dist_coeffs = np.array([k1, k2, p1, p2, k3])
        camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

        frame = cv2.undistort(frame, camera_matrix, dist_coeffs)
    elif filtro == 'blur':
        # Filtro de desenfoque
        # HACER NOSOTROS
        kernel_size = (31, 31)  # Tamaño del kernel de desenfoque
        frame = cv2.GaussianBlur(frame, kernel_size, 0)
        # fractal trace
        # filtro cubism 
    return frame
def main():
    # Argumentos de línea de comandos
    parser = argparse.ArgumentParser(description='Aplicar efectos a una transmisión de cámara en vivo.')
    parser.add_argument('--filtro', type=str, default='contraste', choices=['contraste', 'ecualizacion', 'alien', 'poster', 'distorsion', 'blur'],
                        help='Elegir el filtro a aplicar (contraste, ecualizacion, alien, poster, distorsion, blur)')
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