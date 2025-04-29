import cv2
import numpy as np

def read_and_resize(path, scale=0.5):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"No se encontró la imagen: {path}")
    return cv2.resize(img, (0, 0), fx=scale, fy=scale)

def detect_and_match_features(img1, img2):
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)

    pts1, pts2 = [], []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            pts1.append(kp1[m.queryIdx].pt)
            pts2.append(kp2[m.trainIdx].pt)

    return np.float32(pts1), np.float32(pts2)

def stitch_pair(base_img, new_img):
    pts1, pts2 = detect_and_match_features(base_img, new_img)
    H, _ = cv2.findHomography(pts2, pts1, cv2.RANSAC)

    height1, width1 = base_img.shape[:2]
    height2, width2 = new_img.shape[:2]

    corners_new = np.float32([[0, 0], [0, height2], [width2, height2], [width2, 0]]).reshape(-1, 1, 2)
    warped_corners = cv2.perspectiveTransform(corners_new, H)

    all_corners = np.concatenate((np.float32([[0, 0], [0, height1], [width1, height1], [width1, 0]]).reshape(-1, 1, 2), warped_corners), axis=0)
    [xmin, ymin] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(all_corners.max(axis=0).ravel() + 0.5)

    translation = [-xmin, -ymin]
    H_translate = np.array([[1, 0, translation[0]], [0, 1, translation[1]], [0, 0, 1]])

    result = cv2.warpPerspective(new_img, H_translate @ H, (xmax - xmin, ymax - ymin))
    result[translation[1]:translation[1] + height1, translation[0]:translation[0] + width1] = base_img

    return result

def crop_black_borders(panorama):
    gray = cv2.cvtColor(panorama, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        x, y, w, h = cv2.boundingRect(contours[0])
        return panorama[y:y+h, x:x+w]
    return panorama

# MAIN
if __name__ == "__main__":
    # Rutas de las imágenes en orden
    paths = ["006.jpg", "007.jpg", "008.jpg"]
    images = [read_and_resize(p, 0.5) for p in paths]

    panorama = images[0]
    for i in range(1, len(images)):
        panorama = stitch_pair(panorama, images[i])

   # Solo se une la primera y segunda imagen (la tercera queda sin usar)
    panorama = stitch_pair(images[0], images[1])

    # No se recortan bordes negros
    # No se muestra ni guarda correctamente

    cv2.imshow("Panorama", panorama)
    cv2.waitKey(0)
    cv2.destroyAllWindows()