import os
import cv2
import numpy as np
import time
import sys

def extract_features(image, method, nfeatures=1000):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if method == 'ORB':
        detector = cv2.ORB_create(nfeatures=nfeatures)
    elif method == 'SIFT':
        detector = cv2.SIFT_create(nfeatures=nfeatures)
    elif method == 'AKAZE':
        detector = cv2.AKAZE_create()
    else:
        print("Invalid method. Using ORB by default.")
        detector = cv2.ORB_create(nfeatures=nfeatures)

    keypoints, descriptors = detector.detectAndCompute(gray, None)
    return keypoints, descriptors

def match_features_ratio_test(des1, des2, ratio=0.75):
    des1 = des1.astype(np.uint8)  # Convertir descriptores a tipo CV_8U
    des2 = des2.astype(np.uint8)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(des1, des2, k=2)
    good_matches = []
    for m, n in matches:
        if m.distance < ratio * n.distance:
            good_matches.append(m)
    return good_matches

def main():
    if len(sys.argv) < 2:
        print("No se ha proporcionado ningún parámetro.")
        return
    metodo = sys.argv[1]

    if metodo not in ['ORB', 'SIFT', 'AKAZE']:
        print("Método no válido. Usando ORB por defecto.")
        metodo = 'ORB'

    folder_path = 'BuildingScene'
    image_files = os.listdir(folder_path)

    print("Número de imágenes en la carpeta:", len(image_files))

    nfeatures = 1000
    panorama = None

    for i, image_file in enumerate(image_files):
        print("Procesando imagen:", image_file)
        if image_file.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(folder_path, image_file)
            img = cv2.imread(image_path)

            keypoints, descriptors = extract_features(img, method=metodo, nfeatures=nfeatures)

            # Iterar sobre todas las imágenes restantes para hacer coincidir características y crear el panorama
            for j, other_image_file in enumerate(image_files):
                if j == i:
                    continue  # Evitar comparar la imagen consigo misma

                if other_image_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    other_image_path = os.path.join(folder_path, other_image_file)
                    other_img = cv2.imread(other_image_path)

                    other_keypoints, other_descriptors = extract_features(other_img, method=metodo, nfeatures=nfeatures)

                    good_matches = match_features_ratio_test(descriptors, other_descriptors)

                    if len(good_matches) < 4:
                        continue  # Necesitamos al menos 4 emparejamientos para calcular una transformación

                    src_pts = np.float32([keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                    dst_pts = np.float32([other_keypoints[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

                    # Calcular la matriz de transformación usando la función findHomography
                    M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

                    # Aplicar la transformación a la imagen actual para alinearla con la imagen anterior
                    warped_image = cv2.warpPerspective(other_img, M, (img.shape[1], img.shape[0]))

                    # Ahora, unir las imágenes alineadas
                    if panorama is None:
                        panorama = img.copy()  # La primera imagen se convierte en el panorama inicial
                    panorama = cv2.addWeighted(panorama, 0.5, warped_image, 0.5, 0)

                    # Mostrar el panorama parcial después de agregar cada imagen
                    cv2.imshow("Panorama Parcial", panorama)
                    cv2.waitKey(1000)  # Esperar 1 segundo antes de procesar la siguiente imagen

    # Mostrar el panorama final
    cv2.imshow("Panorama Final", panorama)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
