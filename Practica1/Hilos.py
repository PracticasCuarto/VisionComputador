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