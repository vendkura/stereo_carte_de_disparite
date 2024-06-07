# EPIPOLES FAIT MAIS PAS DISTRIBUER SUR LES DEUX IMAGES

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
    return [m for m in matches if m.distance < max_distance]

def filter_matches_by_horizontal_alignment(keypoints1, keypoints2, matches, max_vertical_distance):
    filtered_matches = []
    for match in matches:
        pt1 = keypoints1[match.queryIdx].pt
        pt2 = keypoints2[match.trainIdx].pt
        if abs(pt1[1] - pt2[1]) < max_vertical_distance:
            filtered_matches.append(match)
    return filtered_matches

def draw_matches(image1, keypoints1, image2, keypoints2, matches):
    return cv2.drawMatches(image1, keypoints1, image2, keypoints2, matches, None, flags=2)

def calculate_fundamental_matrix(matches, keypoints1, keypoints2):
    points1 = np.float32([keypoints1[m.queryIdx].pt for m in matches])
    points2 = np.float32([keypoints2[m.trainIdx].pt for m in matches])
    F, mask = cv2.findFundamentalMat(points1, points2, cv2.FM_RANSAC)
    inliers1 = points1[mask.ravel() == 1]
    inliers2 = points2[mask.ravel() == 1]
    return F, inliers1, inliers2

def draw_epipolar_lines(image1, image2, points1, points2, F):
    lines1 = cv2.computeCorrespondEpilines(points2.reshape(-1, 1, 2), 2, F)
    lines1 = lines1.reshape(-1, 3)
    img1 = image1.copy()
    
    for r in lines1:
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2] / r[1]])
        x1, y1 = map(int, [img1.shape[1], -(r[2] + r[0] * img1.shape[1]) / r[1]])
        img1 = cv2.line(img1, (x0, y0), (x1, y1), color, 1)

    lines2 = cv2.computeCorrespondEpilines(points1.reshape(-1, 1, 2), 1, F)
    lines2 = lines2.reshape(-1, 3)
    img2 = image2.copy()

    for r in lines2:
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2] / r[1]])
        x1, y1 = map(int, [img2.shape[1], -(r[2] + r[0] * img2.shape[1]) / r[1]])
        img2 = cv2.line(img2, (x0, y0), (x1, y1), color, 1)

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

    # Afficher les correspondances
    image_correspondances_final = draw_matches(image_gauche, keypoints_gauche, image_droite, keypoints_droite, matches)
    cv2.imshow('Correspondances Finale', image_correspondances_final)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Afficher les lignes épipolaires
    img1_with_lines, img2_with_lines = draw_epipolar_lines(image_gauche, image_droite, inliers1, inliers2, F)
    cv2.imshow('Image Gauche avec Lignes Épipolaires', img1_with_lines)
    cv2.imshow('Image Droite avec Lignes Épipolaires', img2_with_lines)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
