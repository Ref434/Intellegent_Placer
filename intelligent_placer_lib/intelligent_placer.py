from imageio.core.util import Array
import cv2
import os
import numpy as np
from imageio import imread
from skimage.filters import threshold_otsu
from skimage.morphology import binary_closing,  binary_opening
from scipy.ndimage import binary_fill_holes
from skimage.measure import regionprops
from skimage.measure import label as sk_measure_label
from dataclasses import dataclass
from enum import Enum


class Cases(Enum):
    TRUE = 1
    FALSE = 2
    ERROR = 3


class Config():

    gaussian_sigma: float = 0

    canny_low_threshold: float = 40
    canny_high_threshold: float = 55

    threshold_closing_footprint_width: int = 40
    threshold_closing_footprint = np.ones(
        (threshold_closing_footprint_width, threshold_closing_footprint_width))

    threshold_opening_footprint_width: int = 10
    threshold_opening_footprint = np.ones(
        (threshold_opening_footprint_width, threshold_opening_footprint_width))

    canny_closing_footprint_width: int = 3
    canny_closing_footprint = np.ones(
        (canny_closing_footprint_width, canny_closing_footprint_width))

    canny_opening_footprint_width: int = 5
    canny_opening_footprint = np.ones(
        (canny_opening_footprint_width, canny_opening_footprint_width))

    gaussian_footprint_width: int = 5
    gaussian_footprint = (gaussian_footprint_width, gaussian_footprint_width)

    source_image_path: str = "data\Objects"
    input_image_path: str = "data\InputData"

    false_image: str = "intelligent_placer_lib\\false.jpg"
    error_image: str = "intelligent_placer_lib\\error.png"

    max_angle: int = 360
    step_angle: int = 5
    step_x: int = 5
    step_y: int = 1


@dataclass
class Image():

    name: str = None
    case: Cases = None  # values : True, False, Error
    origin: Array = None
    mask = None
    polygon = None
    result: Cases = None  # values : True, False, Error
    polygon_area: float = None
    items_area: float = None
    items = []
    result_image = None
    test = None



def _find_area(image: Image):
    labels_polygon = sk_measure_label(image.polygon)
    labels_image = sk_measure_label(image.mask)

    props_image = regionprops(labels_image)
    props_polygon = regionprops(labels_polygon)

    image.polygon_area = np.array([prop.area for prop in props_polygon]).sum()
    image.items_area = np.array(
        [prop.area for prop in props_image]).sum() - image.polygon_area


def _polygon_detection(image: Image):

    labels = sk_measure_label(image.mask)
    props = regionprops(labels)

    polygons = np.array([prop.centroid[0] for prop in props])
    index = 0
    if (len(polygons) != 0):
        index = polygons.argmin()

    image.polygon = (labels == (index + 1))


def _custom_key(item):
    label = sk_measure_label(item)
    prop = regionprops(label)
    return prop[0].area


def _items_detection(image: Image):

    bitwise_not_polygon = cv2.bitwise_not(image.polygon.astype("uint8"))
    items = cv2.bitwise_and(image.mask.astype("uint8"), bitwise_not_polygon)

    labels = sk_measure_label(items)
    props = regionprops(labels)

    for index in range(len(props)):
        image.items.append(labels == (index + 1))

    # Sorting by area
    image.items.sort(key = _custom_key, reverse=True)


def processing_source_data():
    config = Config()
    images = []

    for filename in os.listdir(config.source_image_path):
        image = imread(os.path.join(config.source_image_path, filename))

        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_blur_gray = cv2.GaussianBlur(image_gray, (1, 1), config.gaussian_sigma)
        thresh_otsu = threshold_otsu(image_blur_gray)
        res_otsu = image_blur_gray <= thresh_otsu

        res_otsu = binary_closing(res_otsu, footprint = config.threshold_closing_footprint)

        mask = (res_otsu * 255).astype("uint8")
        result = cv2.bitwise_and(image, image, mask=mask)


        images.append([image, res_otsu, result])

    return images


def _read_files(image_path: str):
    image_list = []

    for case in Cases:
        case_folder = os.path.join(image_path, case.name)
        for filename in os.listdir(case_folder):
            if filename.endswith(".jpg"):
                full_path = os.path.join(case_folder, filename)
                image_list.append(_read_one_file(full_path, case))

    return image_list


def _read_one_file(full_path: str, case=None):

    origin_image = imread(full_path)
    name = full_path.split("\\")[-1]
    image = Image(name=name, case=case, origin=origin_image)

    return image


def processing_input_data():
    config = Config()
    images = _read_files(config.input_image_path)

    algorithm_success = 0
    tests_count = 0

    for image in images:
        tests_count += 1
        _image_processing(image, config)

        if _is_error(image):
            error_image = imread(config.error_image)
            image.polygon = error_image
            case = Cases.ERROR
        else:
            case = _fit_in_polygon(image, config)

        if case == image.case:
            algorithm_success += 1

        image.result = case.name

    return images, algorithm_success/tests_count


def _image_processing(image: Image, config: Config):
    image.items = []
    image.item_area = []

    mask = cv2.cvtColor(image.origin, cv2.COLOR_BGR2GRAY)
    mask = cv2.GaussianBlur(mask, config.gaussian_footprint, config.gaussian_sigma)
    mask = cv2.Canny(mask, config.canny_low_threshold, config.canny_high_threshold)

    mask = binary_closing(mask, footprint = config.canny_closing_footprint)
    mask = binary_fill_holes(mask)
    mask = binary_opening(mask, footprint = config.canny_opening_footprint)
    image.mask = mask
    

    _polygon_detection(image)
    _find_area(image)
    _items_detection(image)



def check_image(path: str):
    config = Config()
    image = _read_one_file(path)

    _image_processing(image, config)

    if _is_error(image):
        error_image = imread(config.error_image)
        image.polygon = error_image
        case = Cases.ERROR
    else:
        case = _fit_in_polygon(image, config)

    image.result = case.name

    return image


"""
_rotate(item, angle) - Rotate the item to fit into a polygon
"""
def _rotate(item, angle):
    rotated = item.astype(np.uint8)

    (h, w) = rotated.shape[:2]
    center = (int(w / 2), int(h / 2))

    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)
    rotated = cv2.warpAffine(rotated, rotation_matrix, (w, h))

    return _image_cut(rotated)


"""
_is_error(image: Image) - Checking error cases
"""
def _is_error(image: Image) -> bool:
    y, x = image.mask.shape
    polygon = image.mask[0:y//2, 0:x]
    labels_polygon = sk_measure_label(polygon)
    props_polygon = regionprops(labels_polygon)

    if len(props_polygon) != 1 or image.items_area == 0:
        return True

    return False


"""
_image_cut(image) - Intended for subsequent use in bit operations
"""
def _image_cut(image):
    vertical_indices = np.where(np.any(image, axis=1))[0]
    top, bottom = vertical_indices[0], vertical_indices[-1]

    horizontal_indices = np.where(np.any(image, axis=0))[0]
    left, right = horizontal_indices[0], horizontal_indices[-1]

    return image[top:bottom, left:right]


def _is_item_fit(image: Image, item, config: Config) -> Cases:

    image.polygon = _image_cut(image.polygon)

    polygon_y, polygon_x = image.polygon.shape

    for angle in range(0, config.max_angle, config.step_angle):

        item_rotated = _rotate(item, angle)
        item_y, item_x = item_rotated.shape

        for y in range(0,  polygon_y - item_y, config.step_y):
            for x in range(0, polygon_x - item_x, config.step_x):

                item_contour_box = image.polygon[y: y + item_y, x: x + item_x].astype(int)
                bitwise_and = cv2.bitwise_and(item_contour_box.astype(
                    "uint8"), item_rotated.astype("uint8"))

                if np.sum(bitwise_and) == np.sum(item_rotated):
                    image.polygon[y: y + item_y, x: x + item_x] = cv2.bitwise_xor(
                        item_contour_box.astype("uint8"), item_rotated.astype("uint8"))
                    return Cases.TRUE

    return Cases.FALSE


def _fit_in_polygon(image: Image, config: Config) -> Cases:

    false_image = imread(config.false_image)

    if image.items_area > image.polygon_area:
        image.polygon = false_image
        return Cases.FALSE

    for item in image.items:
        if _is_item_fit(image, item, config) == Cases.FALSE:
            image.polygon = false_image
            return Cases.FALSE

    return Cases.TRUE
