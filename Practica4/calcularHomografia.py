import numpy as np

def findHomographyRANSAC(pts1, pts2, threshold=4.0, max_iterations=1000):
    assert len(pts1) == len(pts2), "Número desigual de puntos"
    assert len(pts1) >= 4, "Se necesitan al menos 4 puntos para calcular la homografía"

    best_inliers = []
    best_H = None
    num_inliers = 0

    for _ in range(max_iterations):
        # Escoger cuatro puntos aleatorios
        indices = np.random.choice(len(pts1), 4, replace=False)
        src_pts = pts1[indices]
        dst_pts = pts2[indices]

        # Calcular la matriz de transformación
        H = computeHomography(src_pts, dst_pts)

        # Calcular la distancia entre puntos transformados y puntos reales
        transformed_pts = applyHomography(pts1, H)
        distances = np.linalg.norm(transformed_pts - pts2, axis=1)

        # Contar los puntos que están dentro del umbral
        inliers = np.where(distances < threshold)[0]

        # Actualizar la mejor homografía si encontramos más inliers
        if len(inliers) > num_inliers:
            best_inliers = inliers
            num_inliers = len(inliers)
            best_H = H

    # Refinar la homografía utilizando todos los inliers encontrados
    refined_H = computeHomography(pts1[best_inliers], pts2[best_inliers])

    return refined_H, best_inliers

def computeHomography(src_pts, dst_pts):
    A = []
    for i in range(len(src_pts)):
        x, y = src_pts[i]
        u, v = dst_pts[i]
        A.append([-x, -y, -1, 0, 0, 0, u*x, u*y, u])
        A.append([0, 0, 0, -x, -y, -1, v*x, v*y, v])
    A = np.asarray(A)
    _, _, V = np.linalg.svd(A)
    H = V[-1].reshape(3, 3)
    H /= H[2, 2]
    return H

def applyHomography(pts, H):
    homogeneous_pts = np.column_stack([pts, np.ones(len(pts))])
    transformed_pts = np.dot(H, homogeneous_pts.T).T
    transformed_pts /= transformed_pts[:, 2][:, np.newaxis]
    return transformed_pts[:, :2]