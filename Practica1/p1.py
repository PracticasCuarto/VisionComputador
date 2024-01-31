import cv2
import numpy as np
import argparse

def aplicar_efectos(frame, filtro):
    if filtro == 'contraste':
        # Efecto de mejora del contraste
        alpha = 1.5  # Factor de aumento de contraste
        beta = 30    # Desplazamiento
        frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)

        frame_gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.equalizeHist(frame_gris)
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    elif filtro == 'alien':
        # Filtro Alien: Cambiar el color de la piel a rojo, verde o azul
        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(frame_hsv, lower_skin, upper_skin)
        frame[mask > 0] = [0, 0, 255]  # Cambiar a color rojo

    elif filtro == 'poster':
        # Filtro Póster: Reducir el número de colores
        num_colores = 8
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = (frame // (256 // num_colores)) * (256 // num_colores)
    elif filtro == 'distorsion':
        # Efecto de distorsión de barril y cojín
        # Los siguientes parámetros son solo ejemplos, ajusta según tus necesidades
        k1, k2, k3 = 0.8, 0.0, 0.0  # Coeficientes de distorsión radial
        p1, p2 = 0.0, 0.0  # Coeficientes de distorsión tangencial
        fx, fy = 500, 500  # Distancia focal en píxeles (ejemplo)
        cx, cy = frame.shape[1] // 2, frame.shape[0] // 2  # Punto principal (centro de la imagen)

        dist_coeffs = np.array([k1, k2, p1, p2, k3])
        camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

        frame = cv2.undistort(frame, camera_matrix, dist_coeffs)
    elif filtro == 'blur':
        # Filtro de desenfoque
        kernel_size = (15, 15)  # Tamaño del kernel de desenfoque
        frame = cv2.GaussianBlur(frame, kernel_size, 0)
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