import cv2
import numpy as np
import imutils

def cart2pol(x, y):
    theta = np.arctan2(y, x)
    rho = np.hypot(x, y)
    return theta, rho


def pol2cart(theta, rho):
    x = rho * np.cos(theta)
    y = rho * np.sin(theta)
    return x, y


def rotate_contour(cnt, angle):
    M = cv2.moments(cnt)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])

    cnt_norm = cnt - [cx, cy]
    
    coordinates = cnt_norm[:, 0, :]
    xs, ys = coordinates[:, 0], coordinates[:, 1]
    thetas, rhos = cart2pol(xs, ys)
    
    thetas = np.rad2deg(thetas)
    thetas = (thetas + angle) % 360
    thetas = np.deg2rad(thetas)
    
    xs, ys = pol2cart(thetas, rhos)
    
    cnt_norm[:, 0, 0] = xs
    cnt_norm[:, 0, 1] = ys

    cnt_rotated = cnt_norm + [cx, cy]
    cnt_rotated = cnt_rotated.astype(np.int32)

    return cnt_rotated



def main() :
    image = cv2.imread("task.jpg")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edged = cv2.Canny(gray, 50, 100)

    cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    mask = np.ones(image.shape[:2], dtype="uint8") * 255


    cnt_rotated1 = rotate_contour(cnts[0],-29)
    mask1 = cv2.drawContours(mask, [cnt_rotated1], -1, 0, -2)
    cnt_rotated2 = rotate_contour(cnts[4],-30)
    mask2 = cv2.drawContours(mask1, [cnt_rotated2], -1, 0, 1)
    cnt_rotated3 = rotate_contour(cnts[2],30)
    mask3 = cv2.drawContours(mask2, [cnt_rotated3], -1, 0, 0)
    cnt_rotated4 = rotate_contour(cnts[11],-150)
    mask4 = cv2.drawContours(mask3, [cnt_rotated4], -1, 0, 1)
    cnt_rotated5 = rotate_contour(cnts[19],-15)
    mask5 = cv2.drawContours(mask4, [cnt_rotated5], -1, 0, 1)
    cnt_rotated6 = rotate_contour(cnts[12],-15)
    mask6 = cv2.drawContours(mask5, [cnt_rotated6], -1, 0, -1)
    cnt_rotated7 = rotate_contour(cnts[14],16)
    mask7 = cv2.drawContours(mask6, [cnt_rotated7], -1, 0, -1)

    cnt_rotated8 = rotate_contour(cnts[21],15)
    cv2.drawContours(mask7, [cnt_rotated8], -1, 0, 0)
        
    cv2.imwrite('rectangle_alignment.jpg',mask)


if '__main__' == __name__:
    main()