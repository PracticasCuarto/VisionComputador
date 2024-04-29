import os
import cv2
import numpy as np
import time
import sys

def create_panorama(image1, image2, matches, keypoints1, keypoints2):
    # Extraer las ubicaciones de los puntos coincidentes en ambas imágenes
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Calcular la homografía entre las imágenes
    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)

    # Obtener las dimensiones de la imagen de salida
    h1, w1 = image1.shape[:2]
    h2, w2 = image2.shape[:2]
    corners1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
    corners2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)

    # Transformar las esquinas de la imagen 2 a la perspectiva de la imagen 1
    corners2_transformed = cv2.perspectiveTransform(corners2, H)

    # Combinar las esquinas de ambas imágenes
    corners = np.concatenate((corners1, corners2_transformed), axis=0)

    # Encontrar los límites del panorama
    [xmin, ymin] = np.int32(corners.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(corners.max(axis=0).ravel() + 0.5)

    # Calcular el desplazamiento necesario para que la imagen 1 esté alineada con el origen
    t = [-xmin, -ymin]
    H_translation = np.array([[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]])

    # Aplicar la transformación a la imagen 2
    panorama = cv2.warpPerspective(image2, H_translation.dot(H), (xmax - xmin, ymax - ymin))
    panorama[t[1]:h1 + t[1], t[0]:w1 + t[0]] = image1  # Superponer la imagen 1 en el panorama

    return panorama


# Paso 1: Captura de imágenes, paso a niveles de gris y extracción de puntos de interés
def extract_features(image, method, nfeatures=1000):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convertir a escala de grises
    if method == 'ORB':
        detector = cv2.ORB_create(nfeatures=nfeatures)
    elif method == 'SIFT':
        detector = cv2.SIFT_create(nfeatures=nfeatures)
    # elif method == 'HARRIS':
    #     detector = cv2.cornerHarris(gray,2,3,0.04)
    #     return detector, None
    elif method == 'AKAZE':
        detector = cv2.AKAZE_create()
    else:
        print("Invalid method. Using ORB by default.")
        detector = cv2.ORB_create(nfeatures=nfeatures)

    keypoints, descriptors = detector.detectAndCompute(gray, None)
    return keypoints, descriptors

# Paso 2: Búsqueda de emparejamientos iniciales por fuerza bruta (todos con todos)
# buscando el vecino más próximo
def match_features_brute_force(des1, des2):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    return matches

# Paso 3: Búsqueda de emparejamientos por fuerza bruta, buscando el vecino más próximo
# y comprobando el ratio al segundo vecino
def match_features_ratio_test(des1, des2, ratio=0.75):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(des1, des2, k=2)
    good_matches = []
    for m, n in matches:
        if m.distance < ratio * n.distance:
            good_matches.append(m)
    return good_matches

# Función para dibujar emparejamientos
def draw_matches(image1, keypoints1, image2, keypoints2, matches):
    # Convertir imágenes a escala de grises
    gray_img1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray_img2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Dibujar los emparejamientos
    result = cv2.drawMatches(gray_img1, keypoints1, gray_img2, keypoints2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imshow("Matches", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Función para dibujar puntos de interés en la imagen
def draw_keypoints(image, keypoints):
    for point in keypoints:
        x, y = int(point[1]), int(point[0])  # Convertir las coordenadas a enteros
        cv2.circle(image, (x, y), 5, (0, 0, 255), -1)  # Dibujar un círculo en el punto
    cv2.imshow("Features Detected with Harris", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Función principal
def main():
    # Obtener el primer parámetro pasado al programa
    if len(sys.argv) < 2:
        print("No se ha proporcionado ningún parámetro.")
        return
    metodo = sys.argv[1]

    if (metodo != 'ORB' and metodo != 'SIFT' and metodo != 'HARRIS' and metodo != 'AKAZE' and metodo != 'SURF'):
        print("Método no válido. Usando ORB por defecto.")
        metodo = 'ORB'

    folder_path = 'BuildingScene' 
    image_files = os.listdir(folder_path)

    print("Number of images in folder:", len(image_files))


    # coger puntos de una imagen y comparar con todos los demas calculando la distancia
    for i, image_file in enumerate(image_files):
        print("Processing image:", image_file)
        if image_file.endswith('.jpg') or image_file.endswith('.jpeg') or image_file.endswith('.png') or image_file.endswith('.JPG'):
            image_path = os.path.join(folder_path, image_file)
            print("Processing image:", image_file)
            img = cv2.imread(image_path)

            # Parámetros para los detectores de puntos de interés
            nfeatures = 1000

            # Paso 1: Extracción de características
            keypoints1, descriptors1 = extract_features(img, method=metodo, nfeatures=nfeatures)

            for _, other_image_file in enumerate(image_files[i+1:]):
                if other_image_file.endswith('.jpg') or other_image_file.endswith('.jpeg') or other_image_file.endswith('.png'):
                    other_image_path = os.path.join(folder_path, other_image_file)
                    print("    Matching with:", other_image_file)
                    other_img = cv2.imread(other_image_path)

                    # Paso 1: Extracción de características de la otra imagen
                    keypoints2, descriptors2 = extract_features(other_img, method=metodo, nfeatures=nfeatures)
                    descriptors1 = descriptors1.astype(np.uint8)
                    descriptors2 = descriptors2.astype(np.uint8)
                    
                    # Paso 2: Búsqueda de emparejamientos por fuerza bruta
                    start_time = time.time()
                    matches = match_features_brute_force(descriptors1, descriptors2)
                    end_time = time.time()
                    print(metodo, " brute force matching time:", end_time - start_time)
                    print("    Number of ", metodo, " matches:", len(matches))

                    # Paso 3: Búsqueda de emparejamientos por ratio test
                    start_time = time.time()
                    good_matches = match_features_ratio_test(descriptors1, descriptors2)
                    end_time = time.time()
                    print(metodo, "    ratio test matching time:", end_time - start_time)
                    print("    Number of ", metodo, " good matches:", len(good_matches))

                    # Dibujar los emparejamientos
                    # draw_matches(img, keypoints1_orb, other_img, keypoints2_orb, matches_orb)
                    draw_matches(img, keypoints1, other_img, keypoints2, matches)


if __name__ == "__main__":
    main()

