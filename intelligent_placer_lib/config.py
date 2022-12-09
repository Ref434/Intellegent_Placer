from enum import Enum
import numpy as np

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
    step_angle: int = 1
    step_x: int = 1
    step_y: int = 1