import cv2
from pathlib import Path

from planescolor import utworz_maske, znajdz_kontury


PLANE_INDEX = 6
PLANE_NAME = f"samolot{PLANE_INDEX:02d}.jpg"
THRESHOLD = 50
THRESHOLD_MODE_OPTIONS = ("gray", "saturation", "combined")
THRESHOLD_MODE = "combined"
CLOSE_ITERATIONS = 1
CLOSE_KERNEL_SIZE = 3
IGNORE_BORDER = False


def dostruj():
    image_path = (
        Path(__file__).resolve().parent
        / "Samoloty"
        / "planes"
        / PLANE_NAME
    )
    image = cv2.imread(str(image_path))
    controls = "Dostrajanie segmentacji"
    preview = "Podglad segmentacji"

    cv2.namedWindow(controls)
    cv2.namedWindow(preview)
    cv2.createTrackbar("Prog", controls, THRESHOLD, 255, lambda _: None)
    cv2.createTrackbar(
        "Kanal (0 jasnosc, 1 nasycenie, 2 polaczone)",
        controls,
        THRESHOLD_MODE_OPTIONS.index(THRESHOLD_MODE),
        len(THRESHOLD_MODE_OPTIONS) - 1,
        lambda _: None,
    )
    cv2.createTrackbar(
        "Iteracje zamykania",
        controls,
        CLOSE_ITERATIONS,
        10,
        lambda _: None,
    )
    cv2.createTrackbar(
        "Promien jadra zamykania",
        controls,
        (CLOSE_KERNEL_SIZE - 1) // 2,
        15,
        lambda _: None,
    )
    cv2.createTrackbar(
        "Odrzuc kontury przy krawedzi",
        controls,
        int(IGNORE_BORDER),
        1,
        lambda _: None,
    )

    while True:
        threshold = cv2.getTrackbarPos("Prog", controls)
        threshold_mode = THRESHOLD_MODE_OPTIONS[
            cv2.getTrackbarPos(
                "Kanal (0 jasnosc, 1 nasycenie, 2 polaczone)", controls
            )
        ]
        close_iterations = cv2.getTrackbarPos(
            "Iteracje zamykania", controls
        )
        close_kernel_size = (
            2 * cv2.getTrackbarPos("Promien jadra zamykania", controls) + 1
        )
        ignore_border = bool(
            cv2.getTrackbarPos("Odrzuc kontury przy krawedzi", controls)
        )

        mask = utworz_maske(
            image,
            threshold,
            threshold_mode,
            close_iterations,
            close_kernel_size,
        )
        contours = znajdz_kontury(mask, image, ignore_border)
        result = image.copy()
        for contour in contours:
            cv2.drawContours(result, [contour], -1, (0, 0, 255), 2)
            moments = cv2.moments(contour)
            centroid = (
                round(moments["m10"] / moments["m00"]),
                round(moments["m01"] / moments["m00"]),
            )
            cv2.circle(result, centroid, 5, (255, 255, 255), cv2.FILLED)

        cv2.putText(
            result,
            f"Obiekty: {len(contours)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )
        cv2.imshow(preview, result)
        cv2.imshow("Maska", mask)
        key = cv2.waitKey(30) & 0xFF
        if key in (ord("q"), 27):
            break

    cv2.destroyAllWindows()
    print(f"PLANE_THRESHOLDS[{PLANE_INDEX}] = {threshold}")
    print(f'THRESHOLD_MODES[{PLANE_INDEX}] = "{threshold_mode}"')
    print(f"CLOSE_ITERATIONS[{PLANE_INDEX}] = {close_iterations}")
    print(f"CLOSE_KERNEL_SIZES[{PLANE_INDEX}] = {close_kernel_size}")
    if ignore_border:
        print(f"BORDER_CONTOUR_IMAGES.add({PLANE_INDEX})")
    else:
        print(f"BORDER_CONTOUR_IMAGES.discard({PLANE_INDEX})")


if __name__ == "__main__":
    dostruj()
