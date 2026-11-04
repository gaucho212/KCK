import cv2
from matplotlib import pyplot as plt
from pathlib import Path
import numpy as np


PLANE_THRESHOLDS = [
    120, 80, 190, 60, 50, 120,
    50, 90, 60, 180, 180, 160,
    200, 60, 80, 60, 90, 100,
]
THRESHOLD_MODES = ["gray"] * 18
THRESHOLD_MODES[6] = "combined"
COMBINED_GRAY_THRESHOLD = 130
CLOSE_ITERATIONS = [
    2, 1, 2, 1, 1, 1,
    1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1,
]
CLOSE_KERNEL_SIZES = {3: 5, 4: 11}
BORDER_CONTOUR_IMAGES = {4}


def utworz_maske(image, threshold, threshold_mode, close_iterations,close_kernel_size):
    if threshold_mode == "gray":
        channel = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mask = cv2.threshold(channel, threshold, 255, cv2.THRESH_BINARY_INV)[1]
    elif threshold_mode == "saturation":
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        channel = hsv[:, :, 1]
        mask = cv2.threshold(channel, threshold, 255, cv2.THRESH_BINARY_INV)[1]
    elif threshold_mode == "combined":
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        saturation = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[:, :, 1]
        gray_mask = cv2.threshold(gray, COMBINED_GRAY_THRESHOLD, 255, cv2.THRESH_BINARY_INV)[1]
        saturation_mask = cv2.threshold(saturation, threshold, 255, cv2.THRESH_BINARY_INV)[1]
        mask = cv2.bitwise_or(gray_mask, saturation_mask)
    else:
        raise ValueError(f"Unknown threshold mode: {threshold_mode}")

    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)

    if close_iterations > 0:
        close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel_size, close_kernel_size))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel, iterations=close_iterations)
    return mask


def znajdz_kontury(mask, image, ignore_border):
    found_contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    height, width = image.shape[:2]
    contours = []
    for contour in found_contours:
        if cv2.contourArea(contour) < 500:
            continue
        x, y, contour_width, contour_height = cv2.boundingRect(contour)
        touches_border = (
            x == 0
            or y == 0
            or x + contour_width >= width
            or y + contour_height >= height
        )
        if ignore_border and touches_border:
            continue
        contours.append(contour)
    return contours


def mozaika(images, thresholds):
    fig, axes = plt.subplots(3, 6, figsize=(18, 6))
    
    i = 0
    for ax in axes.flat:
        image = images[i]
        threshold = thresholds[i]
        threshold_mode = THRESHOLD_MODES[i]
        close_iterations = CLOSE_ITERATIONS[i]
        close_kernel_size = CLOSE_KERNEL_SIZES.get(i, 3)
        
        mask = utworz_maske(image, threshold, threshold_mode, close_iterations, close_kernel_size)
        contours = znajdz_kontury(mask, image, i in BORDER_CONTOUR_IMAGES)

        result = image.copy()
        centroids = []
        for plane_index, contour in enumerate(contours):
            hue = int(179 * plane_index / max(len(contours), 1))
            hsv_color = np.uint8([[[hue, 220, 255]]])
            color = []
            for channel in cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0, 0]:
                color.append(int(channel))
            color = tuple(color)
            moments = cv2.moments(contour)
            centroid = (round(moments["m10"] / moments["m00"]), round(moments["m01"] / moments["m00"]),)
            centroids.append((contour, centroid, color))

        for contour, centroid, color in centroids:
            cv2.drawContours(result, [contour], -1, color, 2)
            radius = max(7, round(min(image.shape[:2]) * 0.012))
            cv2.circle(result, centroid, radius + 2, (0, 0, 0), cv2.FILLED)
            cv2.circle(result, centroid, radius, (255, 255, 255), cv2.FILLED)

        ax.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        ax.axis("off")
        
        i += 1

    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)
    output_path = Path(__file__).resolve().parent / "color_planes_conturs.pdf"
    fig.savefig(output_path)
    plt.show()


def main():
    planes = [
        "samolot00.jpg", "samolot01.jpg", "samolot02.jpg",
        "samolot03.jpg", "samolot04.jpg", "samolot05.jpg",
        "samolot06.jpg", "samolot07.jpg", "samolot08.jpg",
        "samolot09.jpg", "samolot10.jpg", "samolot11.jpg",
        "samolot12.jpg", "samolot13.jpg", "samolot14.jpg",
        "samolot15.jpg", "samolot16.jpg", "samolot17.jpg",
    ]
    images = []

    for plane in planes:
        path = Path(__file__).resolve().parent / "Samoloty" / "planes" / plane
        images.append(cv2.imread(str(path)))

    mozaika(images, PLANE_THRESHOLDS)

main()
