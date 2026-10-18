# FROZEN COPY of Backend/services/image_service.py from the SUPARCO platform.
# Do NOT edit. The experiment must use exactly the operations the platform uses.
# Note: sharpen_image and deblur_image are identical (same 3x3 kernel);
# the experiment uses sharpen_image only and calls it "sharpening".
import cv2
import numpy as np


def denoise_image(input_path, output_path):

    image = cv2.imread(input_path)

    if image is None:
        raise Exception(f"Could not read image: {input_path}")

    denoised = cv2.fastNlMeansDenoisingColored(
        image,
        None,
        10,
        10,
        7,
        21
    )

    cv2.imwrite(output_path, denoised)

    return output_path


def sharpen_image(input_path, output_path):

    image = cv2.imread(input_path)

    if image is None:
        raise Exception(f"Could not read image: {input_path}")

    kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ])

    sharpened = cv2.filter2D(
        image,
        -1,
        kernel
    )

    cv2.imwrite(output_path, sharpened)

    return output_path


def enhance_image(input_path, output_path):

    image = cv2.imread(input_path)

    if image is None:
        raise Exception(f"Could not read image: {input_path}")

    enhanced = cv2.convertScaleAbs(
        image,
        alpha=1.2,
        beta=20
    )

    cv2.imwrite(output_path, enhanced)

    return output_path


def deblur_image(input_path, output_path):

    image = cv2.imread(input_path)

    if image is None:
        raise Exception(f"Could not read image: {input_path}")

    kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ])

    deblurred = cv2.filter2D(
        image,
        -1,
        kernel
    )

    cv2.imwrite(output_path, deblurred)

    return output_path