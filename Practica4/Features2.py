import os
import cv2
import numpy as np
import time
import sys

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

def image_concatenation_direction(H, img_shape, image_file, other_image_file):
    # Coordenadas de los vértices de la imagen de origen
    src_corners = np.array([[0, 0], [0, img_shape[0]], [img_shape[1], img_shape[0]], [img_shape[1], 0]], dtype=np.float32).reshape(-1, 1, 2)

    # Transforma las coordenadas de los vértices de la imagen de origen a las coordenadas de la imagen de destino
    dst_corners = cv2.perspectiveTransform(src_corners, H)

    # Determina si la imagen de origen se superpone a la izquierda o a la derecha de la imagen de destino
    if dst_corners[:, 0, 0].min() < 0:  # Si alguna coordenada x es negativa, la imagen de origen se superpone a la izquierda
        print(f"La imagen de origen {image_file} se superpone a la izquierda de la imagen de destino {other_image_file}.")
        return False
    else:
        print(f"La imagen de origen {image_file} se superpone a la derecha de la imagen de destino {other_image_file}.")
        return True

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

def match_features_ratio_test(des1, des2, ratio=0.80):
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
    cv2.destroyAllWindows()

def calcularHomografia(img1, img2, method, nfeatures=1000):
    keypoints1, descriptors1 = extract_features(img1, method=method, nfeatures=nfeatures)
    keypoints2, descriptors2 = extract_features(img2, method=method, nfeatures=nfeatures)

    good_matches = match_features_ratio_test(descriptors1, descriptors2)

    draw_matches(img1, keypoints1, img2, keypoints2, good_matches)

    if len(good_matches) < 4:
        return None

    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    H, inliers = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)

    return H

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

    nfeatures = 10000
    image_features = []

    imagenes = []

    for i, image_file in enumerate(image_files):
        imagenes.append((image_file,0))
        # print("Procesando imagen:", image_file)
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

                    # Calcular la distancia media a la que estan los puntos de ambas imagenes
                    distances = np.linalg.norm(src_pts - dst_pts, axis=2)
                    mean_distance = np.mean(distances)

                    if H is not None:
                        # print("RANSAC encontró una transformación válida con", len(inliers), "inliers y distancia media", mean_distance)

                        derecha = image_concatenation_direction(H, img.shape, image_file, other_image_file)
                        if derecha:
                            # Derecha
                            image_features.append((other_image_file, image_file, other_img, img, mean_distance))
                        else:
                            # Izquierda
                            image_features.append((image_file, other_image_file, img, other_img, mean_distance))

    numImagenes = 5
    # Ordenar el panorama por orden de distancias
    sorted_features = sorted(image_features, key=lambda x: x[-1])
    num_images_to_keep = numImagenes - 1
    image_count = {}
    final_features = []

    for image_file, other_image_file, img, other_img, distancia in sorted_features:
        # Contar la cantidad de veces que aparece cada imagen
        image_count[image_file] = image_count.get(image_file, 0) + 1
        image_count[other_image_file] = image_count.get(other_image_file, 0) + 1

        # Verificar que ninguna imagen aparezca más de dos veces
        if image_count[image_file] <= 2 and image_count[other_image_file] <= 2:
            # Añadir el par de imágenes a la lista final
            final_features.append((image_file, other_image_file, img, other_img, distancia))

            # Verificar si ya se alcanzó el número deseado de imágenes
            if len(final_features) == num_images_to_keep:
                break
        else:
            # Si una imagen ya apareció dos veces, eliminar todas las parejas que contengan esa imagen
            final_features = [(f1, f2, img1, img2, dist) for f1, f2, img1, img2, dist in final_features
                            if f1 != image_file and f2 != image_file]

    # Imprimir los resultados finales
    for image_file, other_image_file, img, other_img, distancia in final_features:
        print(f"{image_file} - {other_image_file} - Distancia: {distancia}")

    all_other_images = set(other_image_file for _, other_image_file, _, _, _ in final_features)

    # Encontrar la imagen que no se menciona como 'other_image_file'
    first_left_image = None
    for image_file, _, _, _, _ in final_features:
        if image_file not in all_other_images:
            first_left_image = image_file
            break

    print("Imagen izquierda:", first_left_image)

    # TODO: Esta pocho por que habría que mirar cual es la siguiente de verdad pero por casualidad sale bien jeje

    # Crear el panorama empezando por la imagen izquierda y concatenando las demás imágenes
    panorama = None
    for image_file, other_image_file, img, other_img, distancia in final_features:
        if (image_file == first_left_image):
            H = calcularHomografia(img, other_img, metodo, nfeatures)
            
            panorama = warp_images(other_img, img, H)
            # Eliminar de la lista 
            final_features.remove((image_file, other_image_file, img, other_img, distancia))
            cv2.imshow("Panorama", panorama)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    # for image_file, other_image_file, img, other_img, distancia in final_features:
    #     print(f"{image_file} - {other_image_file} - Distancia: {distancia}")
    #     H = calcularHomografia(other_img, panorama, metodo, nfeatures)
    #     panorama = warp_images(panorama, other_img, H)
    #     # Mostrar el panorama final
    #     cv2.imshow("Panorama", panorama)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()

    image_file, other_image_file, img, other_img, distancia = final_features[0]
    print(f"{image_file} - {other_image_file} - Distancia: {distancia}")
    H = calcularHomografia(other_img, panorama, metodo, nfeatures)
    panorama = warp_images(panorama, other_img, H)
    # Mostrar el panorama final
    cv2.imshow("Panorama", panorama)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    image_file, other_image_file, img, other_img, distancia = final_features[1]
    print(f"{image_file} - {other_image_file} - Distancia: {distancia}")
    H = calcularHomografia(panorama, other_img, metodo, nfeatures)
    panorama = warp_images(other_img, panorama, H)
    # Mostrar el panorama final
    cv2.imshow("Panorama", panorama)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    image_file, other_image_file, img, other_img, distancia = final_features[2]
    print(f"{image_file} - {other_image_file} - Distancia: {distancia}")
    H = calcularHomografia(other_img, panorama, metodo, nfeatures)
    panorama = warp_images(panorama, other_img, H)
    # Mostrar el panorama final
    cv2.imshow("Panorama", panorama)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    
    # Mostrar el panorama final
    cv2.imshow("Panorama", panorama)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
