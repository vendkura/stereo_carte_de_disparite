# import cv2

# def load_and_resize_image(path, size):
#     image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
#     return cv2.resize(image, size)

# def detect_sift_features(image):
#     sift = cv2.SIFT_create()
#     keypoints, descriptors = sift.detectAndCompute(image, None)
#     return keypoints, descriptors

# def draw_keypoints(image, keypoints):
#     return cv2.drawKeypoints(image, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# def match_features(descriptors1, descriptors2, ratio):
#     bf = cv2.BFMatcher(cv2.NORM_L2)
#     matches = bf.knnMatch(descriptors1, descriptors2, k=2)
#     good_matches = []
#     for m, n in matches:
#         if m.distance < ratio * n.distance:
#             good_matches.append(m)
#     return good_matches

# def draw_matches(image1, keypoints1, image2, keypoints2, matches):
#     return cv2.drawMatches(image1, keypoints1, image2, keypoints2, matches, None, flags=2)

# def main():
#     size = (600, 800)
#     image_gauche = load_and_resize_image('./images/imgL.jpg', size)
#     image_droite = load_and_resize_image('./images/imgR.jpg', size)

#     keypoints_gauche, descriptors_gauche = detect_sift_features(image_gauche)
#     keypoints_droite, descriptors_droite = detect_sift_features(image_droite)

#     image_gauche_keypoints = draw_keypoints(image_gauche, keypoints_gauche)
#     image_droite_keypoints = draw_keypoints(image_droite, keypoints_droite)

#     cv2.imshow('Points SIFT Gauche', image_gauche_keypoints)
#     cv2.imshow('Points SIFT Droite', image_droite_keypoints)
#     cv2.waitKey(0)

#     matches = match_features(descriptors_gauche, descriptors_droite, 0.4)
#     image_correspondances = draw_matches(image_gauche, keypoints_gauche, image_droite, keypoints_droite, matches)

#     cv2.imshow('Correspondances Filtrées', image_correspondances)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     main()


# import cv2
# import numpy as np

# def load_and_resize_image(path, size):
#     image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
#     return cv2.resize(image, size)


# def detect_sift_features(image):
#     sift = cv2.SIFT_create()
#     keypoints, descriptors = sift.detectAndCompute(image, None)
#     return keypoints, descriptors

# def draw_keypoints(image, keypoints):
#     return cv2.drawKeypoints(image, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# def match_features(descriptors1, descriptors2, ratio):
#     bf = cv2.BFMatcher(cv2.NORM_L2)
#     matches = bf.knnMatch(descriptors1, descriptors2, k=2)
#     good_matches = []
#     for m, n in matches:
#         if m.distance < ratio * n.distance:
#             good_matches.append(m)
#     return good_matches

# def filter_matches_by_distance(matches, max_distance):
#     # Filtrer les correspondances par distance
#     return [m for m in matches if m.distance < max_distance]

# def filter_matches_by_horizontal_alignment(keypoints1, keypoints2, matches, max_vertical_distance):
#     # Filtrer les correspondances pour s'assurer qu'elles sont principalement horizontales
#     filtered_matches = []
#     for match in matches:
#         pt1 = keypoints1[match.queryIdx].pt
#         pt2 = keypoints2[match.trainIdx].pt
#         if abs(pt1[1] - pt2[1]) < max_vertical_distance:
#             filtered_matches.append(match)
#     return filtered_matches

# def draw_matches(image1, keypoints1, image2, keypoints2, matches):
#     return cv2.drawMatches(image1, keypoints1, image2, keypoints2, matches, None, flags=2)

# # Implémentation de la géométrie épipolaire
# def calculate_fundamental_matrix(matches, keypoints1, keypoints2):
#     # Extraire les points des correspondances
#     points1 = np.float32([keypoints1[m.queryIdx].pt for m in matches])
#     points2 = np.float32([keypoints2[m.trainIdx].pt for m in matches])

#     # Calculer la matrice fondamentale
#     F, mask = cv2.findFundamentalMat(points1, points2, cv2.FM_RANSAC)
#     return F, mask

# def draw_epipolar_lines(image1, image2, keypoints1, keypoints2, matches, F):
#     # Sélectionner les points et dessiner les lignes épipolaires sur les deux images
#     lines = cv2.computeCorrespondEpilines(np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2), 2, F)
#     lines = lines.reshape(-1, 3)
#     img1, img2 = image1.copy(), image2.copy()

#     for r, pt1, pt2 in zip(lines, np.float32([keypoints1[m.queryIdx].pt for m in matches]), np.float32([keypoints2[m.trainIdx].pt for m in matches])):
#         color = tuple(np.random.randint(0, 255, 3).tolist())
#         x0, y0 = map(int, [0, -r[2]/r[1]])
#         x1, y1 = map(int, [img1.shape[1], -(r[2]+r[0]*img1.shape[1])/r[1]])
#         pt1 = tuple(map(int, pt1))  # Convert float coordinates to int for cv2.circle
#         pt2 = tuple(map(int, pt2))  # Convert float coordinates to int for cv2.circle
#         img1 = cv2.line(img1, (x0, y0), (x1, y1), color, 1)
#         img1 = cv2.circle(img1, pt1, 5, color, -1)
#         img2 = cv2.circle(img2, pt2, 5, color, -1)

#     return img1, img2


# def main():
#     size = (600, 800)
#     image_gauche = load_and_resize_image('./images/imgL.jpg', size)
#     image_droite = load_and_resize_image('./images/imgR.jpg', size)

#     keypoints_gauche, descriptors_gauche = detect_sift_features(image_gauche)
#     keypoints_droite, descriptors_droite = detect_sift_features(image_droite)

#     matches = match_features(descriptors_gauche, descriptors_droite, 0.8)  # Ajustement du ratio
#     print(f"Nombre de correspondances SIFT: {len(matches)}")
#     F, mask = calculate_fundamental_matrix(matches, keypoints_gauche, keypoints_droite)
#     print("Matrice Fondamentale:", F)

#     # Appliquer les filtres un par un et observer les résultats
#     matches = filter_matches_by_distance(matches, 300)  # Ajustement du seuil de distance
#     matches = filter_matches_by_horizontal_alignment(keypoints_gauche, keypoints_droite, matches, 50)  # Ajustement du seuil d'alignement horizontal
#     print(f"Nombre de correspondances après application des heuristiques: {len(matches)}")

#     image_correspondances_final = draw_matches(image_gauche, keypoints_gauche, image_droite, keypoints_droite, matches)
#     cv2.imshow('Correspondances Finale', image_correspondances_final)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

#     img1_with_lines, img2_with_lines = draw_epipolar_lines(image_gauche, image_droite, keypoints_gauche, keypoints_droite, matches, F)
#     cv2.imshow('Image Gauche avec Lignes Épipolaires', img1_with_lines)
#     cv2.imshow('Image Droite avec Points Correspondants', img2_with_lines)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
    


# if __name__ == "__main__":
#     main()

# EPIPOLES FAIT MAIS PAS DISTRIBUER SUR LES DEUX IMAGES
# import cv2
# import numpy as np

# def load_and_resize_image(path, size):
#     image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
#     return cv2.resize(image, size)

# def detect_sift_features(image):
#     sift = cv2.SIFT_create()
#     keypoints, descriptors = sift.detectAndCompute(image, None)
#     return keypoints, descriptors

# def draw_keypoints(image, keypoints):
#     return cv2.drawKeypoints(image, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# def match_features(descriptors1, descriptors2, ratio):
#     bf = cv2.BFMatcher(cv2.NORM_L2)
#     matches = bf.knnMatch(descriptors1, descriptors2, k=2)
#     good_matches = []
#     for m, n in matches:
#         if m.distance < ratio * n.distance:
#             good_matches.append(m)
#     return good_matches

# def filter_matches_by_distance(matches, max_distance):
#     return [m for m in matches if m.distance < max_distance]

# def filter_matches_by_horizontal_alignment(keypoints1, keypoints2, matches, max_vertical_distance):
#     filtered_matches = []
#     for match in matches:
#         pt1 = keypoints1[match.queryIdx].pt
#         pt2 = keypoints2[match.trainIdx].pt
#         if abs(pt1[1] - pt2[1]) < max_vertical_distance:
#             filtered_matches.append(match)
#     return filtered_matches

# def draw_matches(image1, keypoints1, image2, keypoints2, matches):
#     return cv2.drawMatches(image1, keypoints1, image2, keypoints2, matches, None, flags=2)

# def calculate_fundamental_matrix(matches, keypoints1, keypoints2):
#     points1 = np.float32([keypoints1[m.queryIdx].pt for m in matches])
#     points2 = np.float32([keypoints2[m.trainIdx].pt for m in matches])
#     F, mask = cv2.findFundamentalMat(points1, points2, cv2.FM_RANSAC)
#     inliers1 = points1[mask.ravel() == 1]
#     inliers2 = points2[mask.ravel() == 1]
#     return F, inliers1, inliers2

# def draw_epipolar_lines(image1, image2, points1, points2, F):
#     lines1 = cv2.computeCorrespondEpilines(points2.reshape(-1, 1, 2), 2, F)
#     lines1 = lines1.reshape(-1, 3)
#     img1 = image1.copy()
    
#     for r in lines1:
#         color = tuple(np.random.randint(0, 255, 3).tolist())
#         x0, y0 = map(int, [0, -r[2] / r[1]])
#         x1, y1 = map(int, [img1.shape[1], -(r[2] + r[0] * img1.shape[1]) / r[1]])
#         img1 = cv2.line(img1, (x0, y0), (x1, y1), color, 1)

#     lines2 = cv2.computeCorrespondEpilines(points1.reshape(-1, 1, 2), 1, F)
#     lines2 = lines2.reshape(-1, 3)
#     img2 = image2.copy()

#     for r in lines2:
#         color = tuple(np.random.randint(0, 255, 3).tolist())
#         x0, y0 = map(int, [0, -r[2] / r[1]])
#         x1, y1 = map(int, [img2.shape[1], -(r[2] + r[0] * img2.shape[1]) / r[1]])
#         img2 = cv2.line(img2, (x0, y0), (x1, y1), color, 1)

#     return img1, img2

# def main():
#     size = (600, 800)
#     image_gauche = load_and_resize_image('./images/imgL.jpg', size)
#     image_droite = load_and_resize_image('./images/imgR.jpg', size)

#     keypoints_gauche, descriptors_gauche = detect_sift_features(image_gauche)
#     keypoints_droite, descriptors_droite = detect_sift_features(image_droite)

#     matches = match_features(descriptors_gauche, descriptors_droite, 0.8)
#     print(f"Nombre de correspondances SIFT: {len(matches)}")
    
#     # Filtrage des correspondances
#     matches = filter_matches_by_distance(matches, 300)
#     matches = filter_matches_by_horizontal_alignment(keypoints_gauche, keypoints_droite, matches, 50)
#     print(f"Nombre de correspondances après application des heuristiques: {len(matches)}")
    
#     F, inliers1, inliers2 = calculate_fundamental_matrix(matches, keypoints_gauche, keypoints_droite)
#     print("Matrice Fondamentale:", F)

#     # Afficher les correspondances
#     image_correspondances_final = draw_matches(image_gauche, keypoints_gauche, image_droite, keypoints_droite, matches)
#     cv2.imshow('Correspondances Finale', image_correspondances_final)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

#     # Afficher les lignes épipolaires
#     img1_with_lines, img2_with_lines = draw_epipolar_lines(image_gauche, image_droite, inliers1, inliers2, F)
#     cv2.imshow('Image Gauche avec Lignes Épipolaires', img1_with_lines)
#     cv2.imshow('Image Droite avec Lignes Épipolaires', img2_with_lines)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     main()


# EPIPOLE MIS EN EVIDENCE AVEC LES POINTS CORRESPONDANTS ET LES LIGNES
# import cv2
# import numpy as np

# def load_and_resize_image(path, size):
#     image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
#     return cv2.resize(image, size)

# def detect_sift_features(image):
#     sift = cv2.SIFT_create()
#     keypoints, descriptors = sift.detectAndCompute(image, None)
#     return keypoints, descriptors

# def draw_keypoints(image, keypoints):
#     return cv2.drawKeypoints(image, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# def match_features(descriptors1, descriptors2, ratio):
#     bf = cv2.BFMatcher(cv2.NORM_L2)
#     matches = bf.knnMatch(descriptors1, descriptors2, k=2)
#     good_matches = []
#     for m, n in matches:
#         if m.distance < ratio * n.distance:
#             good_matches.append(m)
#     return good_matches

# def filter_matches_by_distance(matches, max_distance):
#     return [m for m in matches if m.distance < max_distance]

# def filter_matches_by_horizontal_alignment(keypoints1, keypoints2, matches, max_vertical_distance):
#     filtered_matches = []
#     for match in matches:
#         pt1 = keypoints1[match.queryIdx].pt
#         pt2 = keypoints2[match.trainIdx].pt
#         if abs(pt1[1] - pt2[1]) < max_vertical_distance:
#             filtered_matches.append(match)
#     return filtered_matches

# def draw_matches(image1, keypoints1, image2, keypoints2, matches):
#     return cv2.drawMatches(image1, keypoints1, image2, keypoints2, matches, None, flags=2)

# def calculate_fundamental_matrix(matches, keypoints1, keypoints2):
#     points1 = np.float32([keypoints1[m.queryIdx].pt for m in matches])
#     points2 = np.float32([keypoints2[m.trainIdx].pt for m in matches])
#     F, mask = cv2.findFundamentalMat(points1, points2, cv2.FM_RANSAC, 3, 0.99)  # Ajustement des paramètres RANSAC
#     inliers1 = points1[mask.ravel() == 1]
#     inliers2 = points2[mask.ravel() == 1]
#     return F, inliers1, inliers2

# def draw_epipolar_lines(image1, image2, points1, points2, F):
#     lines1 = cv2.computeCorrespondEpilines(points2.reshape(-1, 1, 2), 2, F)
#     lines1 = lines1.reshape(-1, 3)
#     img1 = image1.copy()
    
#     for r in lines1:
#         color = tuple(np.random.randint(0, 255, 3).tolist())
#         x0, y0 = map(int, [0, -r[2] / r[1]])
#         x1, y1 = map(int, [img1.shape[1], -(r[2] + r[0] * img1.shape[1]) / r[1]])
#         img1 = cv2.line(img1, (x0, y0), (x1, y1), color, 1)

#     lines2 = cv2.computeCorrespondEpilines(points1.reshape(-1, 1, 2), 1, F)
#     lines2 = lines2.reshape(-1, 3)
#     img2 = image2.copy()

#     for r in lines2:
#         color = tuple(np.random.randint(0, 255, 3).tolist())
#         x0, y0 = map(int, [0, -r[2] / r[1]])
#         x1, y1 = map(int, [img2.shape[1], -(r[2] + r[0] * img2.shape[1]) / r[1]])
#         img2 = cv2.line(img2, (x0, y0), (x1, y1), color, 1)

#     return img1, img2

# def draw_corresponding_points(image1, image2, points1, points2):
#     img1 = image1.copy()
#     img2 = image2.copy()

#     for pt1, pt2 in zip(points1, points2):
#         color = tuple(np.random.randint(0, 255, 3).tolist())
#         pt1 = tuple(map(int, pt1))
#         pt2 = tuple(map(int, pt2))
#         img1 = cv2.circle(img1, pt1, 5, color, -1)
#         img2 = cv2.circle(img2, pt2, 5, color, -1)

#     return img1, img2

# def main():
#     size = (600, 800)
#     image_gauche = load_and_resize_image('./images/imgL.jpg', size)
#     image_droite = load_and_resize_image('./images/imgR.jpg', size)

#     keypoints_gauche, descriptors_gauche = detect_sift_features(image_gauche)
#     keypoints_droite, descriptors_droite = detect_sift_features(image_droite)

#     matches = match_features(descriptors_gauche, descriptors_droite, 0.8)
#     print(f"Nombre de correspondances SIFT: {len(matches)}")
    
#     # Afficher les correspondances avant filtrage
#     image_correspondances_initial = draw_matches(image_gauche, keypoints_gauche, image_droite, keypoints_droite, matches)
#     cv2.imshow('Correspondances Initiales', image_correspondances_initial)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
    
#     # Filtrage des correspondances
#     matches = filter_matches_by_distance(matches, 300)
#     matches = filter_matches_by_horizontal_alignment(keypoints_gauche, keypoints_droite, matches, 50)
#     print(f"Nombre de correspondances après application des heuristiques: {len(matches)}")
    
#     F, inliers1, inliers2 = calculate_fundamental_matrix(matches, keypoints_gauche, keypoints_droite)
#     print("Matrice Fondamentale:", F)

#     # Afficher les correspondances après filtrage
#     image_correspondances_final = draw_matches(image_gauche, keypoints_gauche, image_droite, keypoints_droite, matches)
#     cv2.imshow('Correspondances Finale', image_correspondances_final)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

#     # Afficher les points correspondants
#     img1_with_points, img2_with_points = draw_corresponding_points(image_gauche, image_droite, inliers1, inliers2)
#     cv2.imshow('Points Correspondants Gauche', img1_with_points)
#     cv2.imshow('Points Correspondants Droite', img2_with_points)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

#     # Afficher les lignes épipolaires
#     img1_with_lines, img2_with_lines = draw_epipolar_lines(image_gauche, image_droite, inliers1, inliers2, F)
#     cv2.imshow('Image Gauche avec Lignes Épipolaires', img1_with_lines)
#     cv2.imshow('Image Droite avec Lignes Épipolaires', img2_with_lines)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     main()


# EPIPOLES FAIT MAIS PAS DISTRIBUER SUR LES DEUX IMAGES
# import cv2
# import numpy as np

# def load_and_resize_image(path, size):
#     image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
#     return cv2.resize(image, size)

# def detect_sift_features(image):
#     sift = cv2.SIFT_create()
#     keypoints, descriptors = sift.detectAndCompute(image, None)
#     return keypoints, descriptors

# def draw_keypoints(image, keypoints):
#     return cv2.drawKeypoints(image, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# def match_features(descriptors1, descriptors2, ratio):
#     bf = cv2.BFMatcher(cv2.NORM_L2)
#     matches = bf.knnMatch(descriptors1, descriptors2, k=2)
#     good_matches = []
#     for m, n in matches:
#         if m.distance < ratio * n.distance:
#             good_matches.append(m)
#     return good_matches

# def filter_matches_by_distance(matches, max_distance):
#     return [m for m in matches if m.distance < max_distance]

# def filter_matches_by_horizontal_alignment(keypoints1, keypoints2, matches, max_vertical_distance):
#     filtered_matches = []
#     for match in matches:
#         pt1 = keypoints1[match.queryIdx].pt
#         pt2 = keypoints2[match.trainIdx].pt
#         if abs(pt1[1] - pt2[1]) < max_vertical_distance:
#             filtered_matches.append(match)
#     return filtered_matches

# def draw_matches(image1, keypoints1, image2, keypoints2, matches):
#     return cv2.drawMatches(image1, keypoints1, image2, keypoints2, matches, None, flags=2)

# def calculate_fundamental_matrix(matches, keypoints1, keypoints2):
#     points1 = np.float32([keypoints1[m.queryIdx].pt for m in matches])
#     points2 = np.float32([keypoints2[m.trainIdx].pt for m in matches])
#     F, mask = cv2.findFundamentalMat(points1, points2, cv2.FM_RANSAC, 3, 0.99)
#     inliers1 = points1[mask.ravel() == 1]
#     inliers2 = points2[mask.ravel() == 1]
#     return F, inliers1, inliers2

# def draw_epipolar_lines(image1, image2, points1, points2, F):
#     lines1 = cv2.computeCorrespondEpilines(points2.reshape(-1, 1, 2), 2, F)
#     lines1 = lines1.reshape(-1, 3)
#     img1 = image1.copy()
    
#     for r in lines1:
#         color = (0, 0, 255)  # Jaune
#         x0, y0 = map(int, [0, -r[2] / r[1]])
#         x1, y1 = map(int, [img1.shape[1], -(r[2] + r[0] * img1.shape[1]) / r[1]])
#         img1 = cv2.line(img1, (x0, y0), (x1, y1), color, 1)

#     lines2 = cv2.computeCorrespondEpilines(points1.reshape(-1, 1, 2), 1, F)
#     lines2 = lines2.reshape(-1, 3)
#     img2 = image2.copy()

#     for r in lines2:
#         color = (0, 0, 255)  # Jaune
#         x0, y0 = map(int, [0, -r[2] / r[1]])
#         x1, y1 = map(int, [img2.shape[1], -(r[2] + r[0] * img2.shape[1]) / r[1]])
#         img2 = cv2.line(img2, (x0, y0), (x1, y1), color, 1)

#     return img1, img2

# def draw_corresponding_points(image1, image2, points1, points2):
#     img1 = image1.copy()
#     img2 = image2.copy()

#     for pt1, pt2 in zip(points1, points2):
#         color = (0, 255, 255)  # Jaune
#         pt1 = tuple(map(int, pt1))
#         pt2 = tuple(map(int, pt2))
#         img1 = cv2.circle(img1, pt1, 10, color, -1)
#         img2 = cv2.circle(img2, pt2, 10, color, -1)

#     return img1, img2

# def main():
#     size = (600, 800)
#     image_gauche = load_and_resize_image('./images/imgL.jpg', size)
#     image_droite = load_and_resize_image('./images/imgR.jpg', size)

#     keypoints_gauche, descriptors_gauche = detect_sift_features(image_gauche)
#     keypoints_droite, descriptors_droite = detect_sift_features(image_droite)

#     matches = match_features(descriptors_gauche, descriptors_droite, 0.8)
#     print(f"Nombre de correspondances SIFT: {len(matches)}")
    
#     # Afficher les correspondances avant filtrage
#     image_correspondances_initial = draw_matches(image_gauche, keypoints_gauche, image_droite, keypoints_droite, matches)
#     cv2.imshow('Correspondances Initiales', image_correspondances_initial)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
    
#     # Filtrage des correspondances
#     matches = filter_matches_by_distance(matches, 300)
#     matches = filter_matches_by_horizontal_alignment(keypoints_gauche, keypoints_droite, matches, 50)
#     print(f"Nombre de correspondances après application des heuristiques: {len(matches)}")
    
#     F, inliers1, inliers2 = calculate_fundamental_matrix(matches, keypoints_gauche, keypoints_droite)
#     print("Matrice Fondamentale:", F)

#     # Afficher les correspondances après filtrage
#     image_correspondances_final = draw_matches(image_gauche, keypoints_gauche, image_droite, keypoints_droite, matches)
#     cv2.imshow('Correspondances Finale', image_correspondances_final)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

#     # Afficher les points correspondants
#     img1_with_points, img2_with_points = draw_corresponding_points(image_gauche, image_droite, inliers1, inliers2)
#     cv2.imshow('Points Correspondants Gauche', img1_with_points)
#     cv2.imshow('Points Correspondants Droite', img2_with_points)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

#     # Afficher les lignes épipolaires
#     img1_with_lines, img2_with_lines = draw_epipolar_lines(image_gauche, image_droite, inliers1, inliers2, F)
#     cv2.imshow('Image Gauche avec Lignes Épipolaires', img1_with_lines)
#     cv2.imshow('Image Droite avec Lignes Épipolaires', img2_with_lines)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     main()


import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_and_resize_image(path, size):
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return cv2.resize(image, size)

def detect_sift_features(image):
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    return keypoints, descriptors

def draw_keypoints(image, keypoints):
    return cv2.drawKeypoints(image, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

def match_features(descriptors1, descriptors2, ratio):
    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)
    good_matches = []
    for m, n in matches:
        if m.distance < ratio * n.distance:
            good_matches.append(m)
    return good_matches

def filter_matches_by_distance(matches, max_distance):
    return [m for m in matches if m.distance < max_distance]

def filter_matches_by_horizontal_alignment(keypoints1, keypoints2, matches, max_vertical_distance):
    filtered_matches = []
    for match in matches:
        pt1 = keypoints1[match.queryIdx].pt
        pt2 = keypoints2[match.trainIdx].pt
        if abs(pt1[1] - pt2[1]) < max_vertical_distance:
            filtered_matches.append(match)
    return filtered_matches

def calculate_fundamental_matrix(matches, keypoints1, keypoints2):
    points1 = np.float32([keypoints1[m.queryIdx].pt for m in matches])
    points2 = np.float32([keypoints2[m.trainIdx].pt for m in matches])
    F, mask = cv2.findFundamentalMat(points1, points2, cv2.FM_RANSAC, 3, 0.99)
    inliers1 = points1[mask.ravel() == 1]
    inliers2 = points2[mask.ravel() == 1]
    return F, inliers1, inliers2

def draw_corresponding_lines(image1, image2, points1, points2):
    img1 = cv2.cvtColor(image1, cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(image2, cv2.COLOR_GRAY2BGR)
    
    for pt_left, pt_right in zip(points1, points2):
        x1, y1 = pt_left
        x2, y2 = pt_right
        cv2.line(img1, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
        cv2.circle(img1, (int(x1), int(y1)), 5, (0, 255, 255), -1)
        cv2.circle(img2, (int(x2), int(y2)), 5, (0, 255, 255), -1)
    
    return img1, img2

def main():
    size = (600, 800)
    image_gauche = load_and_resize_image('./images/imgL.jpg', size)
    image_droite = load_and_resize_image('./images/imgR.jpg', size)

    keypoints_gauche, descriptors_gauche = detect_sift_features(image_gauche)
    keypoints_droite, descriptors_droite = detect_sift_features(image_droite)

    matches = match_features(descriptors_gauche, descriptors_droite, 0.8)
    print(f"Nombre de correspondances SIFT: {len(matches)}")

    # Filtrage des correspondances
    matches = filter_matches_by_distance(matches, 300)
    matches = filter_matches_by_horizontal_alignment(keypoints_gauche, keypoints_droite, matches, 50)
    print(f"Nombre de correspondances après application des heuristiques: {len(matches)}")

    F, inliers1, inliers2 = calculate_fundamental_matrix(matches, keypoints_gauche, keypoints_droite)
    print("Matrice Fondamentale:", F)

    # Afficher les correspondances après filtrage
    pts_left = [keypoints_gauche[m.queryIdx].pt for m in matches]
    pts_right = [keypoints_droite[m.trainIdx].pt for m in matches]
    img1_with_lines, img2_with_lines = draw_corresponding_lines(image_gauche, image_droite, pts_left, pts_right)

    # Afficher les résultats
    plt.figure(figsize=(20, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(img1_with_lines, cv2.COLOR_BGR2RGB))
    plt.title('Correspondances SIFT sur Image Gauche')

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(img2_with_lines, cv2.COLOR_BGR2RGB))
    plt.title('Correspondances SIFT sur Image Droite')

    plt.show()

if __name__ == "__main__":
    main()
