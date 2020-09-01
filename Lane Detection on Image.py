import cv2
import numpy as np
import matplotlib.pyplot as plt

def reg_of_interest(img, vertices):
    mask = np.zeros_like(img)
    # channels = img.shape[2]
    match_mask_color = 255
    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

img = cv2.imread('data/lane.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
height, width = img.shape[0], img.shape[1]

region_of_interest = [(0, height), ((width/2)+70, (height/2)+50), (width, height)]

gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
canny_img = cv2.Canny(gray, 150, 200)
masked_img = reg_of_interest(canny_img, np.array([region_of_interest], np.int32))
lines = cv2.HoughLinesP(masked_img, rho=6, theta=np.pi/180, threshold=100, lines=np.array([]),
                        minLineLength=50, maxLineGap=10)

for line in lines:
    x1, y1, x2, y2 = line[0]
    cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 3)

plt.imshow(img)
plt.show()
