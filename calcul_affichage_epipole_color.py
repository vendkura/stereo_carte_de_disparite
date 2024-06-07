
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
