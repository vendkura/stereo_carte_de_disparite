##################################
# Dans ce code, nous affichons : 
# - Recherche des points SIFT
# - Les correspondances initiales
# - Les correspondances finales
##################################
import cv2
import numpy as np

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
    # Filtrer les correspondances par distance
    return [m for m in matches if m.distance < max_distance]

def filter_matches_by_horizontal_alignment(keypoints1, keypoints2, matches, max_vertical_distance):
    # Filtrer les correspondances pour s'assurer qu'elles sont principalement horizontales
    filtered_matches = []
    for match in matches:
        pt1 = keypoints1[match.queryIdx].pt
        pt2 = keypoints2[match.trainIdx].pt
        if abs(pt1[1] - pt2[1]) < max_vertical_distance:
            filtered_matches.append(match)
    return filtered_matches

def draw_matches(image1, keypoints1, image2, keypoints2, matches):
    return cv2.drawMatches(image1, keypoints1, image2, keypoints2, matches, None, flags=2)

# Implémentation de la géométrie épipolaire
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


def main():
    size = (600, 800)
    image_gauche = load_and_resize_image('./images/imgL.jpg', size)
    image_droite = load_and_resize_image('./images/imgR.jpg', size)

    keypoints_gauche, descriptors_gauche = detect_sift_features(image_gauche)
    keypoints_droite, descriptors_droite = detect_sift_features(image_droite)

    matches = match_features(descriptors_gauche, descriptors_droite, 0.8)  # Ajustement du ratio
    print(f"Nombre de correspondances SIFT: {len(matches)}")
    # F, mask = calculate_fundamental_matrix(matches, keypoints_gauche, keypoints_droite)
    # print("Matrice Fondamentale:", F)

    # Appliquer les filtres un par un et observer les résultats
    matches = filter_matches_by_distance(matches, 300)  # Ajustement du seuil de distance
    matches = filter_matches_by_horizontal_alignment(keypoints_gauche, keypoints_droite, matches, 50)  # Ajustement du seuil d'alignement horizontal
    print(f"Nombre de correspondances après application des heuristiques: {len(matches)}")


    
    image_gauche_keypoints = draw_keypoints(image_gauche, keypoints_gauche)
    image_droite_keypoints = draw_keypoints(image_droite, keypoints_droite)
    cv2.imshow('Points SIFT Gauche', image_gauche_keypoints)
    cv2.imshow('Points SIFT Droite', image_droite_keypoints)
    cv2.waitKey(0)

    image_correspondances = draw_matches(image_gauche, keypoints_gauche, image_droite, keypoints_droite, matches)
    cv2.imshow('Correspondances Filtrées', image_correspondances)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    image_correspondances_final = draw_matches(image_gauche, keypoints_gauche, image_droite, keypoints_droite, matches)
    cv2.imshow('Correspondances Finale', image_correspondances_final)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    


if __name__ == "__main__":
    main()


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
