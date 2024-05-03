import os
import cv2
import numpy as np
import time
import sys

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
        if filename.endswith('.jpg'):
            print(f"Procesando imagen: {filename}")
            img = cv2.imread(os.path.join(folder_path, filename))
            imagenes.append(img)
    return imagenes

def imagenVaDerecha(imagen1, imagen2):
    # Coordenadas de los vértices de la imagen de origen
    src_corners = np.array([[0, 0], [0, imagen1.shape[0]], [imagen1.shape[1], imagen1.shape[0]], [imagen1.shape[1], 0]], dtype=np.float32).reshape(-1, 1, 2)

    # Calcular  la homografía entre las dos imágenes
    GoodMatches, BaseImage_kp, SecImage_kp = EncontrarMatches(imagen1, imagen2)

    # Calcular la matriz de homografía
    H = calcularHomografia(BaseImage_kp, SecImage_kp, GoodMatches)

    # Transforma las coordenadas de los vértices de la imagen de origen a las coordenadas de la imagen de destino
    dst_corners = cv2.perspectiveTransform(src_corners, H)

    # Determina si la imagen de origen se superpone a la izquierda o a la derecha de la imagen de destino
    if dst_corners[:, 0, 0].min() < 0:  # Si alguna coordenada x es negativa, la imagen de origen se superpone a la izquierda
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
                if imagenVaDerecha(imagenes[i], imagenes[j]):
                    # Si la imagen j está a la derecha de la imagen i, aumentamos el contador de la imagen i
                    izquierda_count[i] += 1

    # Ordenar las imágenes según el número de imágenes a su izquierda
    sorted_indices = sorted(izquierda_count, key=lambda x: izquierda_count[x])

    # Ordenar la lista de imágenes según el orden determinado
    imagenes_ordenadas = [imagenes[i] for i in sorted_indices]
    print(f"Orden de las imágenes: {sorted_indices}")
    return imagenes_ordenadas


# Define el orden de las imagenes devolviendo la imagen central y las listas de imagenes de la izquierda y derecha
# Esto lo hemos hecho para que el resultado no quede distorsionado
def dividirImagenes(imagenes):
    # Definir la mitad de la lista de imágenes
    n_mitad = len(imagenes) // 2

    mitad = imagenes[n_mitad]
    izquierda = []
    derecha = []

    # Definir las imágenes de la izquierda
    for i in range(n_mitad):
        izquierda.append(imagenes[i])

    # Definir las imágenes de la derecha
    for i in range(n_mitad + 1, len(imagenes)):
        derecha.append(imagenes[i])

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
    # If less than 4 matches found, exit the code.
    if len(GoodMatches) < 4:
        print("\nNot enough matches found between the images.\n")
        exit(0)

    # Storing coordinates of points corresponding to the matches found in both the images
    BaseImage_pts = []
    SecImage_pts = []
    for Match in GoodMatches:
        BaseImage_pts.append(BaseImage_kp[Match[0].queryIdx].pt)
        SecImage_pts.append(SecImage_kp[Match[0].trainIdx].pt)

    # Changing the datatype to "float32" for finding homography
    BaseImage_pts = np.float32(BaseImage_pts)
    SecImage_pts = np.float32(SecImage_pts)

    # Finding the homography matrix(transformation matrix).
    (HomographyMatrix, _) = cv2.findHomography(SecImage_pts, BaseImage_pts, cv2.RANSAC, 4.0)

    return HomographyMatrix


def GetNewFrameSizeAndMatrix(HomographyMatrix, Sec_ImageShape, Base_ImageShape):
    # Reading the size of the image
    (Height, Width) = Sec_ImageShape
    
    # Taking the matrix of initial coordinates of the corners of the secondary image
    # Stored in the following format: [[x1, x2, x3, x4], [y1, y2, y3, y4], [1, 1, 1, 1]]
    # Where (xt, yt) is the coordinate of the i th corner of the image. 
    InitialMatrix = np.array([[0, Width - 1, Width - 1, 0],
                              [0, 0, Height - 1, Height - 1],
                              [1, 1, 1, 1]])
    
    # Finding the final coordinates of the corners of the image after transformation.
    # NOTE: Here, the coordinates of the corners of the frame may go out of the 
    # frame(negative values). We will correct this afterwards by updating the 
    # homography matrix accordingly.
    FinalMatrix = np.dot(HomographyMatrix, InitialMatrix)

    [x, y, c] = FinalMatrix
    x = np.divide(x, c)
    y = np.divide(y, c)

    # Finding the dimentions of the stitched image frame and the "Correction" factor
    min_x, max_x = int(round(min(x))), int(round(max(x)))
    min_y, max_y = int(round(min(y))), int(round(max(y)))

    New_Width = max_x
    New_Height = max_y
    Correction = [0, 0]
    if min_x < 0:
        New_Width -= min_x
        Correction[0] = abs(min_x)
    if min_y < 0:
        New_Height -= min_y
        Correction[1] = abs(min_y)
    
    # Again correcting New_Width and New_Height
    # Helpful when secondary image is overlaped on the left hand side of the Base image.
    if New_Width < Base_ImageShape[1] + Correction[0]:
        New_Width = Base_ImageShape[1] + Correction[0]
    if New_Height < Base_ImageShape[0] + Correction[1]:
        New_Height = Base_ImageShape[0] + Correction[1]

    # Finding the coordinates of the corners of the image if they all were within the frame.
    x = np.add(x, Correction[0])
    y = np.add(y, Correction[1])
    OldInitialPoints = np.float32([[0, 0],
                                   [Width - 1, 0],
                                   [Width - 1, Height - 1],
                                   [0, Height - 1]])
    NewFinalPonts = np.float32(np.array([x, y]).transpose())

    # Updating the homography matrix. Done so that now the secondary image completely 
    # lies inside the frame
    HomographyMatrix = cv2.getPerspectiveTransform(OldInitialPoints, NewFinalPonts)
    
    return [New_Height, New_Width], Correction, HomographyMatrix

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

def UnirImagen(imagenBase, imagenUnir):

    # Encontrar los puntos clave y emparejarlos
    GoodMatches, BaseImage_kp, SecImage_kp = EncontrarMatches(imagenBase, imagenUnir)

    # Calcular la matriz de homografía
    H = calcularHomografia(BaseImage_kp, SecImage_kp, GoodMatches)

    # Actualizar el nuevo tamaño del tablero
    # NewFrameSize, Correction, H = GetNewFrameSizeAndMatrix(H, imagenUnir.shape, imagenBase.shape)

    # Unir las imágenes
    # SecImage_Transformed = cv2.warpPerspective(imagenUnir, H, (NewFrameSize[1], NewFrameSize[0]))
    # BaseImage_Transformed = np.zeros((NewFrameSize[0], NewFrameSize[1], 3), dtype=np.uint8)
    # BaseImage_Transformed[Correction[1]:Correction[1]+imagenBase.shape[0], Correction[0]:Correction[0]+imagenBase.shape[1]] = imagenBase

    # StitchedImage = cv2.bitwise_or(SecImage_Transformed, cv2.bitwise_and(BaseImage_Transformed, cv2.bitwise_not(SecImage_Transformed_Mask)))
    StitchedImage = warp_images(imagenBase, imagenUnir, H)

    return StitchedImage

def main():
    if len(sys.argv) < 2:
        print("No se ha proporcionado ningún parámetro.")
        return
    metodo = sys.argv[1]

    if metodo not in ['ORB', 'SIFT', 'AKAZE']:
        print("Método no válido. Usando ORB por defecto.")
        metodo = 'ORB'

    folder_path = 'Edificio' 

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
    
    cv2.imwrite("Panorama.png", Panorama)


if __name__ == "__main__":
    main()
