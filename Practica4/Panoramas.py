import os
import cv2
import numpy as np
import time
import sys

from calcularHomografia import findHomographyRANSAC

''' 
    Programa que genera el panorama a partir de una secuencia de imágenes cortadas
    tomadas desde un mismo punto rotando la cámara.

    El proceso se divide en los siguientes pasos:
    1. Leer las imágenes
    2. Detectar puntos clave y descriptores
    3. Emparejar los puntos clave
    4. Calcular la homografía
    5. Establecer el orden en el que se van a unir las imágenes
    6. Crear el panorama

'''

# def leerImagenes(folder_path):
#     imagenes = []
#     for filename in os.listdir(folder_path):
#         if filename.endswith('.jpg'):
#             img = cv2.imread(os.path.join(folder_path, filename))
#             imagenes.append(img)
#     return imagenes

def leerImagenes(folder_path):
    imagenes = []
    # Obtener la lista de archivos en el directorio y ordenarla alfabéticamente
    filenames = sorted(os.listdir(folder_path))
    for filename in filenames:
        if filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.JPG'):
            img = cv2.imread(os.path.join(folder_path, filename))
            imagenes.append((img, filename))
    return imagenes

def imagenVaDerecha(imagen1, imagen2):
    # Coordenadas de los vértices de la imagen de origen
    esquinas_src = np.array([[0, 0], [0, imagen1.shape[0]], [imagen1.shape[1], imagen1.shape[0]], [imagen1.shape[1], 0]], dtype=np.float32).reshape(-1, 1, 2)

    # Calcular la homografía entre las dos imágenes
    buenos_emparejamientos, puntos_clave_imagen1, puntos_clave_imagen2 = EncontrarMatches(imagen1, imagen2)

    # Calcular la matriz de homografía
    H = calcularHomografia(puntos_clave_imagen1, puntos_clave_imagen2, buenos_emparejamientos)

    # Transformar las coordenadas de los vértices de la imagen de origen a las coordenadas de la imagen de destino
    esquinas_dst = cv2.perspectiveTransform(esquinas_src, H)

    # Determinar si la imagen de origen se superpone a la izquierda o a la derecha de la imagen de destino
    if esquinas_dst[:, 0, 0].min() < 0:  # Si alguna coordenada x es negativa, la imagen de origen se superpone a la izquierda
        # print(f"La imagen de origen se superpone a la izquierda de la imagen de destino.")
        return False
    else:
        # print(f"La imagen de origen se superpone a la derecha de la imagen de destino.")
        return True


def OrdenarImagenes(imagenes):
    # Crear un diccionario para almacenar el número de imágenes a la izquierda de cada imagen
    izquierda_count = {i: 0 for i in range(len(imagenes))}

    # Comparar cada par de imágenes para determinar la posición relativa
    for i in range(len(imagenes)):
        for j in range(len(imagenes)):
            if i != j:  # No comparamos una imagen consigo misma
                if imagenVaDerecha(imagenes[i][0], imagenes[j][0]):
                    # Si la imagen j está a la derecha de la imagen i, aumentamos el contador de la imagen i
                    izquierda_count[i] += 1

    # Ordenar las imágenes según el número de imágenes a su izquierda
    sorted_indices = sorted(izquierda_count, key=lambda x: izquierda_count[x])

    # Ordenar la lista de imágenes según el orden determinado
    imagenes_ordenadas = [imagenes[i][0] for i in sorted_indices]
    indices_ordenados = [imagenes[i][1] for i in sorted_indices]
    print(f"Orden de las imágenes: {indices_ordenados}")
    return imagenes_ordenadas


# Define el orden de las imagenes devolviendo la imagen central y las listas de imagenes de la izquierda y derecha
# Esto lo hemos hecho para que el resultado no quede distorsionado
def dividirImagenes(imagenes):
    print(len(imagenes))
    # Definir la mitad de la lista de imágenes
    n_mitad = len(imagenes) // 2
    mitad = imagenes[n_mitad]
    izquierda = []
    derecha = []


    # Definir las imágenes de la izquierda
    for i in range(0, n_mitad):
        izquierda.append(imagenes[i])
        print(f"Izquierda: ", i)

    # Definir las imágenes de la derecha
    for i in range(n_mitad + 1, len(imagenes)):
        derecha.append(imagenes[i])
        print(f"Derecha: ", i)

    return mitad, izquierda, derecha

def EncontrarMatches(imagenBase, imagenUnir):
    #  TODO: Uso sift de momento
    Sift = cv2.SIFT_create()
    BaseImage_kp, BaseImage_des = Sift.detectAndCompute(cv2.cvtColor(imagenBase, cv2.COLOR_BGR2GRAY), None)
    SecImage_kp, SecImage_des = Sift.detectAndCompute(cv2.cvtColor(imagenUnir, cv2.COLOR_BGR2GRAY), None)

    # Usar Fuerza Bruta para emparejar los puntos clave
    MatcherFB = cv2.BFMatcher()
    InitialMatches = MatcherFB.knnMatch(BaseImage_des, SecImage_des, k=2)

    # Filtrar los emparejamientos para quedarnos con los buenos (ratio test)
    GoodMatches = []
    for m, n in InitialMatches:
        if m.distance < 0.75 * n.distance:
            GoodMatches.append([m])

    return GoodMatches, BaseImage_kp, SecImage_kp

def calcularHomografia(BaseImage_kp, SecImage_kp, GoodMatches):
    # Si se encuentran menos de 4 emparejamientos, salir del código.
    if len(GoodMatches) < 4:
        print("\nNo se encontraron suficientes emparejamientos entre las imágenes.\n")
        exit(0)

    # Almacenando las coordenadas de los puntos correspondientes a los emparejamientos encontrados en ambas imágenes
    PuntosImagenBase = []
    PuntosImagenSec = []
    for Match in GoodMatches:
        PuntosImagenBase.append(BaseImage_kp[Match[0].queryIdx].pt)
        PuntosImagenSec.append(SecImage_kp[Match[0].trainIdx].pt)

    # Cambiando el tipo de datos a "float32" para encontrar la homografía
    PuntosImagenBase = np.float32(PuntosImagenBase)
    PuntosImagenSec = np.float32(PuntosImagenSec)

    # Encontrar la matriz de homografía (matriz de transformación).
    (MatrizHomografia, _) = findHomographyRANSAC(PuntosImagenSec, PuntosImagenBase, 4.0, 10000)

    return MatrizHomografia


# Funcion obtenida de: https://github.com/PacktPublishing/OpenCV-with-Python-By-Example/tree/master
# Dadas dos imagenes y su homografía calcula la imagen resultante de unirlas
def warp_images(img1, img2, H):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    corners1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
    corners2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
    warped_corners2 = cv2.perspectiveTransform(corners2, H)

    corners = np.concatenate((corners1, warped_corners2), axis=0)
    [xmin, ymin] = np.int32(corners.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(corners.max(axis=0).ravel() + 0.5)

    t = [-xmin, -ymin]
    Ht = np.array([[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]])

    warped_img2 = cv2.warpPerspective(img2,Ht @ H, (xmax - xmin, ymax - ymin))
    warped_img2[t[1]:h1 + t[1], t[0]:w1 + t[0]] = img1

    return warped_img2

# Dadas dos imagenes, une ambas en una 
def UnirImagen(imagen_base, imagen_unir):

    # Encontrar los puntos clave y emparejarlos
    buenos_emparejamientos, PC_base, PC_unir = EncontrarMatches(imagen_base, imagen_unir)

    # Calcular la matriz de homografía
    H = calcularHomografia(PC_base, PC_unir, buenos_emparejamientos)

    # Unir las imágenes
    imagen_unida = warp_images(imagen_base, imagen_unir, H)

    return imagen_unida


def main():
    if len(sys.argv) < 3:
        print("Ejecutar como python Panoramas.py <RutaCarpetaImagenes>, <rutaResultado>")
        return
    folder_path = sys.argv[1]

    res_path = sys.argv[2]

    # Leer las imágenes y definir el orden en el que se van a unir
    imagenes = leerImagenes(folder_path)

    # Establecer el orden a partir de sus homografias
    imagenes = OrdenarImagenes(imagenes)

    # Dividir las imagenes en tres listas: izquierda, derecha y la central para que la union no sea distorsionada
    Panorama, izquierda, derecha = dividirImagenes(imagenes)

    for i in range (0, len(izquierda)):
        # Unir la imagen
        ImagenUnida = UnirImagen(Panorama, izquierda[i])

        Panorama = ImagenUnida.copy()

    for i in range (0, len(derecha)):
        # Unir la imagen
        ImagenUnida = UnirImagen(Panorama, derecha[i])

        Panorama = ImagenUnida.copy()
    
    cv2.imwrite(res_path + ".png", Panorama)


if __name__ == "__main__":
    main()
