import cv2
import numpy as np

def reg_of_interest(img, vertices):
    mask = np.zeros_like(img)
    # channels = img.shape[2]
    match_mask_color = 255
    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

cap = cv2.VideoCapture('data/Lane Video.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    height, width = frame.shape[0], frame.shape[1]
    region_of_interest = [(0, height), ((width/2), (height/2)+160), (width, height)]

    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    canny_frame = cv2.Canny(gray, 150, 200)
    masked_frame = reg_of_interest(canny_frame, np.array([region_of_interest], np.int32))
    lines = cv2.HoughLinesP(masked_frame, rho=6, theta=np.pi/180, threshold=100, lines=np.array([]),
                            minLineLength=50, maxLineGap=10)

    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)

    cv2.imshow('frame', frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
