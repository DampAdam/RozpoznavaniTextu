import cv2
import time
import os
import functools as ft

clr = lambda:os.system('cls')

clr()

c = 0
if not os.path.exists('images'):
    print("Making directory images")
    os.makedirs('images')
else:
    c = ft.reduce(lambda x, y: x if x > y else y, [int(file.split('.')[0]) for file in os.listdir('images')], 0)+1
    print(f"Images already exists, starting from {c}")


cap = cv2.VideoCapture(0)


def captureImages(n=100, imagesPerSec=10):
    waitTime = 1/imagesPerSec
    for i in range(c, n+c):
        _, frame = cap.read()

        cv2.imshow('frame', frame)

        cv2.imwrite(f'images/{i}.jpg', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        time.sleep(waitTime)


for i in range(3):
    print(f"Capturing images in {3-i} seconds")
    time.sleep(1)
    clr()
print("Smile :)")
captureImages(10, 2)
clr()
print("Images captured successfully!")