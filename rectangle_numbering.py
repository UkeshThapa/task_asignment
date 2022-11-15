import cv2
import numpy as np
import os
import imutils


def is_contour_bad(c):
	# approximate the contour
	peri = cv2.arcLength(c, True)
	approx = cv2.approxPolyDP(c, 0.02 * peri, True)
	# the contour is 'bad' if it is not a rectangle
	return not len(approx) == 4

def load_image(img_name):
    image = cv2.imread(img_name)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image,gray

def save_img(img,name=None):
    if name is None:
        cv2.imwrite('result.jpg',img)
    else:
        cv2.imwrite(f'{name}',img)


def find_contour(image, gray):
    edged = cv2.Canny(gray, 50, 100)
    cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)


    return cnts


def main():
    image , gray = load_image('task.jpg')
    cnts= find_contour(image,gray)
    mask = np.ones(image.shape[:2], dtype="uint8") * 255
    # loop over the contours
    for c in cnts:
    # if the contour is bad, draw it on the mask
        if is_contour_bad(c):
            cv2.drawContours(mask, [c], -1, 0, -1)
    save_img(mask,name='mask.jpg')
    img, gry = load_image('mask.jpg')
    os.remove('mask.jpg')

    edges = cv2.Canny(gry, 50, 100, apertureSize=3)
    base = cv2.HoughLinesP(edges, 1, np.pi / 180, 40, minLineLength=1, maxLineGap=6)
    pixel_array = []
    co_ord = []
    pixel_length = 0
    if base is not None:
        for line in base:
            x1, y1, x2, y2 = line[0]
            pixel_length = np.abs(x2 - x1)
            pixel_array.append(pixel_length)
            co_ord.append({'x1' : x1,'y1':y1,'len' : pixel_length})
    
    pixel_array.sort()
    co_ord.pop(-1)
    co = list(set(pixel_array))
    co.sort()
    co.pop(-1)
    for num,i in enumerate(co):
        for j in co_ord:
            if i == j['len']:
                cv2.putText(image, f"{num+1}", (j['x1']+10,j['y1']+55), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 1)
                    # print(j['x1'])
    save_img( image, name = "rectangle_numbering.jpg")


if '__main__' == __name__:
    main()