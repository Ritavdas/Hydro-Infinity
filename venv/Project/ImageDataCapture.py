import time

import cv2

cap = cv2.VideoCapture()
count = 1

name = "Mark"

while (True):

    ret, frame = cap.read()
    time.sleep(1)
    cv2.imwrite(
        "C:\\Users\Ritav\PycharmProjects\Hydro-Infinity\\venv\Project\ImagesDataset\{}\img{}.png".format(name, count),
        frame)

    count = count + 1

    cv2.imshow('Captured Frame', frame)

    if cv2.waitKey(20) & 0xFF == ord('q'):
        break
