# python3 Features2.py SIFT
import os
import cv2
import numpy as np
import time
import sys

folder_path = 'BuildingScene' 

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

    warped_img2 = cv2.warpPerspective(img2, Ht @ H, (xmax - xmin, ymax - ymin))
    warped_img2[t[1]:h1 + t[1], t[0]:w1 + t[0]] = img1

    return warped_img2

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

def match_features_ratio_test(des1, des2, ratio=0.82):
    des1 = des1.astype(np.uint8)  # Convertir descriptores a tipo CV_8U
    des2 = des2.astype(np.uint8)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(des1, des2, k=2)
    good_matches = []
    for m, n in matches:
        if m.distance < ratio * n.distance:
            good_matches.append(m)
    return good_matches


def draw_matches(image1, keypoints1, image2, keypoints2, matches):
    gray_img1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray_img2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    result = cv2.drawMatches(gray_img1, keypoints1, gray_img2, keypoints2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imshow("Matches", result)
    cv2.waitKey(0)
    cv2.destroyAllWindoqws()

def create_panorama(image_files, image_features):
    # Ordenar imágenes por número de inliers
    image_features.sort(key=lambda x: x[2], reverse=True)
    
    # Usar la imagen con más inliers como imagen base
    base_idx, _, _, base_H = image_features[0]
    base_image_file = image_files[base_idx]
    base_image_path = os.path.join(folder_path, base_image_file)
    base_img = cv2.imread(base_image_path)

    # Calcular el número de imagenes en el panorama
    num_images = len(image_files)
    
    # Crear el lienzo para el panorama en este caso horizontal
    panorama_width = base_img.shape[1]
    panorama_height = base_img.shape[0]
    
    for idx, _, _, H in image_features:
        if idx == base_idx:
            continue
        
        image_file = image_files[idx]
        image_path = os.path.join(folder_path, image_file)
        img = cv2.imread(image_path)
        
        # Obtener la imagen transformada
        warped_img = warp_images(img, base_img, H)
        
        # Actualizar las dimensiones del panorama si es necesario
        h, w = warped_img.shape[:2]
        panorama_width = max(panorama_width, w)
        panorama_height = max(panorama_height, h)
    
    # Crear el lienzo del panorama
    panorama = np.zeros((panorama_height, panorama_width, 3), dtype=np.uint8)
    
    # Aplicar las transformaciones y fusionar las imágenes
    for idx, _, _, H in image_features:
        image_file = image_files[idx]
        image_path = os.path.join(folder_path, image_file)
        img = cv2.imread(image_path)
        
        warped_img = warp_images(img, base_img, H)
        
        # Superponer la imagen transformada en el lienzo del panorama
        panorama = blend_images(panorama, warped_img)
    
    # Mostrar el panorama final
    cv2.imshow("Panorama", panorama)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def blend_images(background, foreground):
    # Asegurar que ambas imágenes tengan las mismas dimensiones
    if background.shape[:2] != foreground.shape[:2]:
        foreground = cv2.resize(foreground, (background.shape[1], background.shape[0]))
    
    # Asegurar que ambas imágenes tengan el mismo número de canales
    if background.shape[2] != foreground.shape[2]:
        foreground = cv2.cvtColor(foreground, cv2.COLOR_GRAY2BGR)  # Convertir a color si es necesario

    # Mezclar las imágenes con un peso igual
    blended_image = cv2.addWeighted(background, 0.5, foreground, 0.5, 0)
    return blended_image


def main():
    if len(sys.argv) < 2:
        print("No se ha proporcionado ningún parámetro.")
        return
    metodo = sys.argv[1]

    if metodo not in ['ORB', 'SIFT', 'AKAZE']:
        print("Método no válido. Usando ORB por defecto.")
        metodo = 'ORB'

    
    image_files = os.listdir(folder_path)

    print("Número de imágenes en la carpeta:", len(image_files))

    nfeatures = 10000
    image_features = []

    for i, image_file in enumerate(image_files):
        print("Procesando imagen:", image_file)
        if image_file.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(folder_path, image_file)
            img = cv2.imread(image_path)

            keypoints1, descriptors1 = extract_features(img, method=metodo, nfeatures=nfeatures)

            for _, other_image_file in enumerate(image_files[i+1:]):
                if other_image_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    other_image_path = os.path.join(folder_path, other_image_file)
                    other_img = cv2.imread(other_image_path)

                    keypoints2, descriptors2 = extract_features(other_img, method=metodo, nfeatures=nfeatures)

                    good_matches = match_features_ratio_test(descriptors1, descriptors2)
                    
                    if len(good_matches) < 4:
                        continue  # Necesitamos al menos 4 emparejamientos para calcular una transformación

                    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

                    # Uso de RANSAC para estimar la homografía
                    # H, inliers = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 500.0)
                    H, inliers = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)

                    if H is not None:
                        print("RANSAC encontró una transformación válida con", len(inliers), "inliers.")

                        image_features.append((i, i + 1, len(inliers), H))  # Almacena el índice de las imágenes y el número de inliers

                        print("Imagen 1:", image_file)
                        print("Imagen 2:", other_image_file)
                        result = warp_images(other_img, img, H)

                        # Display the blended image
                        # cv2.imshow('Blended Image', result)
                        # cv2.waitKey(0)
                        # cv2.destroyAllWindows()

    print("Número de emparejamientos válidos:", len(image_features))
    # Ordenar imágenes por número de inliers
    image_features.sort(key=lambda x: x[2], reverse=True)

    print("Número de imágenes con homografía válida:", len(image_features))

    # Crear el panorama
    create_panorama(image_files, image_features)



if __name__ == "__main__":
    main()
