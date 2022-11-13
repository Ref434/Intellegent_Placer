from imageio.core.util import Array
import cv2
import os
import math
import numpy as np
from imageio import imread
from skimage.color import rgb2gray
from skimage.feature import canny
from skimage.filters import gaussian, threshold_otsu
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

    gaussian_sigma_threshold: float = 3.0
    gaussian_sigma_canny: float = 1.5

    canny_sigma: float = 0.3
    canny_low_threshold: float = 0.01
    canny_high_threshold: float = 0.45

    binary_closing_footprint_width: int = 5
    binary_closing_footprint = np.ones((binary_closing_footprint_width, binary_closing_footprint_width))

    image_width: int = 960
    image_height: int = 1280

    source_image_path = "data\Objects"
    input_image_path = "data\InputData"




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
    fit = None
    items = []
    result_image = None
    test = []

      

def get_central_component(mask, config):

    labels = sk_measure_label(mask) 
    props = regionprops(labels) 

    centers = [prop.centroid for prop in props]

    difference = math.sqrt((config.image_height/2)**2
                     + (config.image_width/2)**2)

    i = 0
    for center in centers:
        new_difference = math.sqrt((config.image_height/2 - center[0])**2
                             + (config.image_width/2 - center[1])**2)

        if(new_difference < difference):
            central_comp_id = i
            difference = new_difference

        i += 1

    return labels == (central_comp_id + 1)

def find_area(image: Image):
    labels_polygon = sk_measure_label(image.polygon)
    labels_image  = sk_measure_label(image.mask)

    props_image  = regionprops(labels_image)
    props_polygon = regionprops(labels_polygon)

    image.polygon_area = np.array([prop.area for prop in props_polygon]).sum()
    image.items_area= np.array([prop.area for prop in props_image]).sum() - image.polygon_area

    return image


def polygon_detection(image: Image):

    labels = sk_measure_label(image.mask) 
    props = regionprops(labels)

    polygons = np.array([prop.centroid[0] for prop in props])
    index = 0
    if(len(polygons) != 0):
        index = polygons.argmin()

    image.polygon = (labels == (index + 1))

    return image

def items_detection(image: Image):

    bitwise_not_polygon = cv2.bitwise_not(image.polygon.astype("uint8"))
    items = cv2.bitwise_and(image.mask.astype("uint8"), bitwise_not_polygon)

    labels = sk_measure_label(items) 
    props  = regionprops(labels)
    for index in range(len(props)):
        image.items.append(labels == (index + 1))

    return image



def processing_source_data():
    config = Config()
    images = []

    for filename in os.listdir(config.source_image_path):
        image = imread(os.path.join(config.source_image_path, filename))  
        image_blur_gray = rgb2gray(gaussian(image, sigma = config.gaussian_sigma_threshold, channel_axis = True))
        
        thresh_otsu = threshold_otsu(image_blur_gray)
        res_otsu = image_blur_gray <= thresh_otsu

        res_otsu = binary_closing(res_otsu, footprint=np.ones((40, 40)))
        res_otsu = binary_opening(res_otsu, footprint=np.ones((10, 10)))
        res_otsu = get_central_component(res_otsu, config)


        mask = (res_otsu * 255).astype("uint8")
        result = cv2.bitwise_and(image, image, mask = mask)


        vertical_indices = np.where(np.any(res_otsu, axis=1))[0]
        top, bottom = vertical_indices[0], vertical_indices[-1]

        horizontal_indices = np.where(np.any(res_otsu, axis=0))[0]
        left, right = horizontal_indices[0], horizontal_indices[-1]

        crop_mask = res_otsu[top:bottom, left:right]

        
        images.append([image, crop_mask, result])
        
    return images

def read_files(image_path: str):
    image_list = []

    for case in Cases:
        for filename in os.listdir(f"{image_path}\\{case.name}"):
            if filename.endswith(".jpg"):
                origin_image = imread(os.path.join(f"{image_path}\\{case.name}", filename))
                image = Image(name = filename, case = case, origin = origin_image)
                image_list.append(image)    

    return image_list



def processing_input_data():
    config = Config()
    images = read_files(config.input_image_path)

    algorithm_success = 0
    tests_count = 0

    for image in images:
        image.items = []
        image.test= []
        image.item_area = []
        tests_count += 1

        image_blur_gray = rgb2gray(gaussian(image.origin, sigma = config.gaussian_sigma_canny, channel_axis= True))
        mask = binary_closing(
            canny(
                image_blur_gray,
                sigma = config.canny_sigma,
                low_threshold = config.canny_low_threshold,
                high_threshold = config.canny_high_threshold,
            ),
            footprint = config.binary_closing_footprint
        )
        mask = binary_fill_holes(mask)
        mask = binary_opening(mask, footprint=np.ones((15, 15)))

        image.mask = mask    
        image = polygon_detection(image)
        image = find_area(image)
        image = items_detection(image)

        if is_error(image):
            error_image = imread(os.path.join("intelligent_placer_lib", "error.png"))
            image.polygon = error_image
            case = Cases.ERROR
        else:
            case = fit_in_polygon(image)

        if case == image.case:
            algorithm_success += 1

        image.result = case.name
      
    return images, algorithm_success/tests_count

def rotate(image, angle):
    rotated = image.astype(np.uint8)
    (h, w) = rotated.shape[:2]
    center = (int(w / 2), int(h / 2))
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)
    rotated  = cv2.warpAffine(rotated , rotation_matrix, (w, h))
    vertical_indices = np.where(np.any(rotated , axis=1))[0]
    top, bottom = vertical_indices[0], vertical_indices[-1]
    horizontal_indices = np.where(np.any(rotated , axis=0))[0]
    left, right = horizontal_indices[0], horizontal_indices[-1]
    rotated = rotated[top:bottom, left:right]

    return rotated


def is_error(image: Image):
    y, x = image.mask.shape
    polygon = image.mask[0:y//2+50, 0:x]
    labels_polygon = sk_measure_label(polygon)
    props_polygon = regionprops(labels_polygon)

    if len(props_polygon) != 1 or image.items_area == 0:
        return True


def is_item_fit(image: Image, item):
    vertical_indices = np.where(np.any(image.polygon, axis=1))[0]
    top_p, bottom_p = vertical_indices[0], vertical_indices[-1]

    horizontal_indices = np.where(np.any(image.polygon, axis=0))[0]
    left_p, right_p = horizontal_indices[0], horizontal_indices[-1]

    image.polygon = image.polygon[top_p:bottom_p, left_p:right_p]

    polygon_y, polygon_x = image.polygon.shape

    for angle in range(100, 360, 5):

        item_rotated  = rotate(item, angle)
        item_y, item_x = item_rotated.shape

        for y in range(0,  polygon_y - item_y, 1):
            for x in range(0, polygon_x - item_x, 5):

                item_contour_box = image.polygon[y: y + item_y, x: x + item_x].astype(int)
                bitwise_and = cv2.bitwise_and(item_contour_box.astype("uint8"), item_rotated.astype("uint8"))

                if np.sum(bitwise_and) == np.sum(item_rotated):
                    image.polygon[y: y + item_y, x: x + item_x] = cv2.bitwise_xor(item_contour_box.astype("uint8"), item_rotated.astype("uint8"))
                    return Cases.TRUE

    return Cases.FALSE



def fit_in_polygon(image: Image):

    false_image = imread(os.path.join("intelligent_placer_lib", "false.jpg"))

    if image.items_area > image.polygon_area:
        image.polygon = false_image
        return Cases.FALSE

    for item in image.items:   
        if is_item_fit(image, item) == Cases.FALSE:
            image.polygon = false_image
            return Cases.FALSE
          
    return Cases.TRUE

