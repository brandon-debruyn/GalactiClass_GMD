from common_imports import *

class GalactiClass_PreProcessing(object):
    @staticmethod
    def preprocess_image(image):
        # Image preprocessing code
        pass

    @staticmethod
    def another_preprocessing_function(image):
        # Additional preprocessing steps
        pass
    
    @staticmethod
    def crop_to_galaxy(galaxy_mask, image):
        contours, _ = cv2.findContours(galaxy_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        cropped_image = image[y:y+h, x:x+w]

        return cropped_image
    