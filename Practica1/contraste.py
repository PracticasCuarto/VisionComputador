import cv2
import numpy as np

def aplicar_efectos(frame):
    # Efecto de mejora del contraste
    alpha = 1.5  # Factor de aumento de contraste
    beta = 30    # Desplazamiento

    frame_contraste = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)

    # Efecto de ecualización de histograma
    frame_gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_ecualizado = cv2.equalizeHist(frame_gris)
    frame_ecualizado = cv2.cvtColor(frame_ecualizado, cv2.COLOR_GRAY2BGR)

    # Mostrar los resultados en pantalla
    cv2.imshow('Original', frame)
    cv2.imshow('Contraste Mejorado', frame_contraste)
    cv2.imshow('Ecualización de Histograma', frame_ecualizado)

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
    aplicar_efectos(frame)

    # Salir del bucle si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la cámara y cerrar las ventanas
captura.release()
cv2.destroyAllWindows()
