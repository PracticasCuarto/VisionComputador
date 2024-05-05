import numpy as np
import cv2

def calcularHomografiaRANSAC(pts1, pts2, umbral=4.0, max_iteraciones=1000):
    assert len(pts1) == len(pts2), "Número desigual de puntos"
    assert len(pts1) >= 4, "Se necesitan al menos 4 puntos para calcular la homografía"

    mejores_inliers = []
    num_inliers = 0

    for _ in range(max_iteraciones):
        # Escoger cuatro puntos aleatorios
        indices = np.random.choice(len(pts1), 4, replace=False)
        src_pts = pts1[indices]
        dst_pts = pts2[indices]

        # Calcular la matriz de transformación
        H = calcularHomografia(src_pts, dst_pts)

        # Calcular la distancia entre puntos transformados y puntos reales
        puntos_transformados = aplicarHomografia(pts1, H)
        distancias = np.linalg.norm(puntos_transformados - pts2, axis=1)

        # Contar los puntos que están dentro del umbral
        inliers = np.where(distancias < umbral)[0]

        # Actualizar la mejor homografía si encontramos más inliers
        if len(inliers) > num_inliers:
            mejores_inliers = inliers
            num_inliers = len(inliers)
       
    # Refinar la homografía utilizando todos los inliers encontrados
    H_refinada = calcularHomografia(pts1[mejores_inliers], pts2[mejores_inliers])

    return H_refinada, mejores_inliers

def calcularHomografia(pts_fuente, pts_destino):
    A = []
    for i in range(len(pts_fuente)):
        x, y = pts_fuente[i]
        u, v = pts_destino[i]
        A.append([-x, -y, -1, 0, 0, 0, u*x, u*y, u])
        A.append([0, 0, 0, -x, -y, -1, v*x, v*y, v])
    A = np.asarray(A)
    _, _, V = np.linalg.svd(A)
    H = V[-1].reshape(3, 3)
    H /= H[2, 2]
    return H

def aplicarHomografia(pts, H):
    pts_homogeneos = np.column_stack([pts, np.ones(len(pts))])
    puntos_transformados = np.dot(H, pts_homogeneos.T).T
    puntos_transformados /= puntos_transformados[:, 2][:, np.newaxis]
    return puntos_transformados[:, :2]
