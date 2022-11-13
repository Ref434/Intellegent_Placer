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

class Image():

    name: str = None
    case: str = None  # values : True, False, Error
    origin = None
    mask = None
    polygon = None
    result: str = None  # values : True, False, Error

    def __init__(self, image_name: str, image_case: str, image):

        self.name = image_name
        self.case = image_case
        self.origin = image

        

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

def polygon_detection(mask):

    labels = sk_measure_label(mask) 
    props = regionprops(labels) 

    polygons = np.array([prop.centroid[0] for prop in props])
    index = 0
    if(len(polygons) != 0):
        index = polygons.argmin()

    return labels == (index + 1)


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

    true = "True"
    false = "False"
    error = "Error"

    for filename in os.listdir(image_path + "\\" + true):
        if filename.endswith(".jpg"):
            origin_image = imread(os.path.join(image_path + "\\" + true, filename))
            image = Image(filename, true, origin_image) 
            image_list.append(image)    


    for filename in os.listdir(image_path + "\\" + false):
        if filename.endswith(".jpg"):
            origin_image = imread(os.path.join(image_path + "\\" + false, filename))
            image = Image(filename, false, origin_image) 
            image_list.append(image)  


    for filename in os.listdir(image_path + "\\" + error):
        if filename.endswith(".jpg"):
            origin_image = imread(os.path.join(image_path + "\\" + error, filename))
            image = Image(filename, error, origin_image) 
            image_list.append(image)  
    
    return image_list



def processing_input_data():
    config = Config()
    images = read_files(config.input_image_path)

    placement_success = 0
    tests_count = 0

    for image in images:
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
        image.polygon = polygon_detection(mask)

        fit = fit_in_polygon(image.polygon, image.mask)

        if(fit == image.case):
            image.result = "True"
            placement_success += 1

        else:
            image.result = "False"
      
    return images, placement_success/tests_count

def fit_in_polygon(polygon_mask, image_mask):
    labels_image  = sk_measure_label(image_mask)
    labels_polygon = sk_measure_label(polygon_mask) 

    props_image  = regionprops(labels_image )
    props_polygon = regionprops(labels_polygon) 

    area_image = np.array([prop.area for prop in props_image]).sum()
    area_polygon = np.array([prop.area for prop in props_polygon]).sum()

    if(area_image - 2* area_polygon > 0):
        return "False"
       
    return "True"

