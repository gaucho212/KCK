import cv2
from matplotlib import pyplot as plt
from pathlib import Path
import numpy as np


def mozaika(images, thresholds, modes):
    fig, axes = plt.subplots(2, 3, figsize=(10, 6))
    i = 0
    for ax in axes.flat:
        if modes[i] == "saturation":
            hsv = cv2.cvtColor(images[i], cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv[:, :, 1], 0, thresholds[i])
        else:
            gray = cv2.cvtColor(images[i], cv2.COLOR_BGR2GRAY)
            mask = cv2.threshold(gray, thresholds[i], 255, cv2.THRESH_BINARY_INV)[1]
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        plane_contour = max(contours, key=cv2.contourArea)
        result = np.zeros_like(images[i])
        cv2.drawContours(result, [plane_contour], -1, (255, 255, 255), 2)
        ax.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        ax.axis('off')
        i += 1

    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)
    fig.savefig('blac_white_planes_conturs.pdf')    
    plt.show()


def main():
    planes = ["samolot06.jpg", "samolot08.jpg", "samolot09.jpg", "samolot10.jpg", "samolot12.jpg", "samolot17.jpg"]
    thresholds = [50, 60, 180, 180, 200, 100]
    modes = ["saturation", "brightness", "brightness", "brightness", "brightness", "brightness"]
    images = []
    
    for plane in planes:
        path = Path(__file__).resolve().parent / "Samoloty" / "planes" / plane
        image = cv2.imread(path)
        images.append(image)

    mozaika(images, thresholds, modes)

main()
