from common_imports import *
from helpers import GalacticClass_Helpers
from image_preprocessor import GalactiClass_PreProcessing

class GalactiClass_MorphologyDetector(object):
    """ 
    Class for detecting galaxy morphology and calculating confidence scores. 
    """
    # Threshold definitions
    ELLIPTICAL_THRESHOLD = {
        'ellipticity': 7,
        'r_squared_brightness_profile': 0.85,
        'gini_coefficient': 0.74,
        'asymmetry_index': 1.14,
        'color_gradient': 50
    }
    
    SPIRAL_THRESHOLD = {}
    IRREGULAR_THRESHOLD = {
        'ellipticity': 5,
        'texture_skewness': 1.05,
        'texture_kurtosis': 1.5,
        'solidity': 0.9
    }

    def __init__(self):
        self.helpers = GalacticClass_Helpers()
        self.image_processing = GalactiClass_PreProcessing()
        self.classes = ['elliptical', 'spiral', 'irregular']
        self.class_confidence = {cls: 0.0 for cls in self.classes}

    def _check_ellipticity(self, hubble_ellipticity):
        """
            Check if the hubble ellipticity is within the threshold for elliptical galaxies.

            :param hubble_ellipticity: The calculated hubble ellipticity of the galaxy.
            :return: 1 if the ellipticity is within the threshold, otherwise 0.
        """
        return 1 if hubble_ellipticity is not None and hubble_ellipticity <= self.ELLIPTICAL_THRESHOLD['ellipticity'] else 0

    def _check_brightness_profile(self, brightness_profile):
        """ Check if the brightness profile matches elliptical galaxy criteria. """
        fitted_params = self.helpers.fit_brightness_profile(brightness_profile)
        if fitted_params is not None:
            I0, a = fitted_params
            r_squared = self.helpers.calculate_r_squared(np.array(brightness_profile), fitted_params)
            print(r_squared)
            return 1 if r_squared > self.ELLIPTICAL_THRESHOLD['r_squared_brightness_profile'] else 0
        return 0

    def _check_gini_coefficient(self, gini_coefficient):
        """ Check if the gini coefficient matches elliptical galaxy criteria. """
        return 1 if gini_coefficient > self.ELLIPTICAL_THRESHOLD['gini_coefficient'] else 0

    def _check_asymmetry_index(self, asymmetry_index):
        """ Check if the asymmetry index matches elliptical galaxy criteria. """
        return 1 if asymmetry_index < self.ELLIPTICAL_THRESHOLD['asymmetry_index'] else 0

    def _check_color_gradients(self, color_gradients):
        """ Check if the color gradients match elliptical galaxy criteria. """
        return 1 if color_gradients[0] < self.ELLIPTICAL_THRESHOLD['color_gradient'] else 0       

    def _check_irregular_ellipticity(self, hubble_ellipticity):
        """ Check if the ellipticity matches irregular galaxy criteria."""
        return 1 if hubble_ellipticity is not None and hubble_ellipticity >= self.IRREGULAR_THRESHOLD['ellipticity'] else 0
    
    def _get_elliptical_confidence(self, image) -> (float):
        """
        Calculate the confidence that a given image is an elliptical galaxy.

        :param image: The image of the galaxy.
        :return: The confidence percentage that the galaxy is elliptical.
        """
        
        image = image.copy()
        
        RGB_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        gray_img = cv2.cvtColor(RGB_img, cv2.COLOR_RGB2GRAY)
        
        # segment the mask of the entire galaxy object then crop to galaxy passing image and galaxy mask
        T, focus_galaxy_mask = cv2.threshold(gray_img, 50, 255, cv2.THRESH_BINARY)
        cropped_image = self.image_processing.crop_to_galaxy(focus_galaxy_mask, RGB_img)
        
        # convert cropped image to gray scale and segment nucleus of galaxy
        gray_cropped_img = cv2.cvtColor(cropped_image, cv2.COLOR_RGB2GRAY)
        _, galactic_nucleus_mask = cv2.threshold(gray_cropped_img, 175, 255, cv2.THRESH_BINARY)
        
        # find all the contours thresholded and get largest contour from array of contours
        contours, _ = cv2.findContours(galactic_nucleus_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        nucleus_contour = self.helpers.get_largest_contour(contours)
        
        # calculate the ellipticity of the galaxy using the nucleus contour and get the array of brightness radii of the galaxy
        centroid_x, centroid_y, hubble_ellipticity = self.helpers.get_galaxy_ellipticity(cropped_image, nucleus_contour)
        color_brightness_profiles = self.helpers.calculate_color_brightness_profile(cropped_image, centroid_x=centroid_x, centroid_y=centroid_y)
        
        # calculate the gini coefficient, assymtetry index, and color gradients of the image
        gini_coefficient = self.helpers.calculate_gini_coefficient(cropped_image)
        asymmetry_index = self.helpers.calculate_asymmetry_index_cont(cropped_image)
        color_gradients = self.helpers.calculate_color_gradients(cropped_image)
        
        # return confidence based on ELLIPTICAL THRESHOLD Definition
        confidence = 0
        
        confidence += (1 if self.helpers.elliptical_brightness_profile(color_brightness_profiles['b']) else 0)
        confidence += self._check_ellipticity(hubble_ellipticity)
        confidence += self._check_brightness_profile(color_brightness_profiles['b'])        
        confidence += self._check_gini_coefficient(gini_coefficient)
        confidence += self._check_asymmetry_index(asymmetry_index)
        confidence += self._check_color_gradients(color_gradients)
                
        print(f' elliptical confidence = {confidence} / {len(self.ELLIPTICAL_THRESHOLD.keys()) + 1} ({(confidence / (len(self.ELLIPTICAL_THRESHOLD.keys()) + 1)) * 100 })')
        
        return (confidence / 6.0) * 100
        #return cropped_image # testing
        
    def _get_spiral_confidence(self, image):
        
        # Global thresholds
        threshold_bulge_size = 1000
        threshold_std_dev = 10
        threshold_blue_ratio = 0.70
        
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        contours, _ = cv2.findContours(gray_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        galaxy_contour = self.helpers.get_largest_contour(contours)

        ellipticity, ellipse = self.helpers.calculate_ellipticity(galaxy_contour)
        blue_ratio, blue_image = self.helpers.color_analysis(image, galaxy_contour)
        bulge_size, bulge_image = self.helpers.locate_and_measure_bulge(image, galaxy_contour)
        radial_profile_values = self.helpers.calculate_radial_profile(gray_image, ellipse[0])
        is_likely_spiral_intensity = self.helpers.analyze_intensity_profile(radial_profile_values, threshold_std_dev)
        asymmetry_index_value = self.helpers.asymmetry_index(gray_image)

        # Conditions and their respective likelihood contributions
        conditions = [
            (ellipticity > 0.1, 20),
            (blue_ratio > 0.05 and blue_ratio < threshold_blue_ratio, 20),
            (bulge_size > threshold_bulge_size, 20),
            (is_likely_spiral_intensity, 20),
            (2 < asymmetry_index_value < 3, 20) 
        ]

        # Calculate the likelihood percentage
        likelihood_percentage = sum(percent for condition, percent in conditions if condition)

        return likelihood_percentage
    
    def _get_irregular_confidence(self, image):
        image_cpy = image.copy()

        RGB_img = cv2.cvtColor(image_cpy, cv2.COLOR_BGR2RGB)
        gray_img = cv2.cvtColor(RGB_img, cv2.COLOR_RGB2GRAY)

        T, focus_galaxy_mask = cv2.threshold(gray_img, 50, 255, cv2.THRESH_BINARY)
        cropped_image = self.image_processing.crop_to_galaxy(focus_galaxy_mask, RGB_img)
        
        gray_img = cv2.cvtColor(cropped_image, cv2.COLOR_RGB2GRAY)
        _, thresh = cv2.threshold(gray_img, 175, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        largest_contour = self.helpers.get_largest_contour(contours)
        texture_skewness, texture_kurtosis = self.helpers.calculate_kurtosis_skew(gray_img, thresh)

        _, _, hubble_ellipticity = self.helpers.get_galaxy_ellipticity(cropped_image, largest_contour)

        solidity = self.helpers.calculate_solidity(largest_contour)
        # For Testing
        # Print each feature value
        #print(f"Ellipticity: {hubble_ellipticity}, Skewness: {texture_skewness}, Kurtosis: {texture_kurtosis}, Solidity: {solidity}")

        confidence = 0
        confidence += self._check_irregular_ellipticity(hubble_ellipticity)
        confidence += (1 if texture_skewness > self.IRREGULAR_THRESHOLD['texture_skewness'] else 0)
        confidence += (1 if texture_kurtosis < self.IRREGULAR_THRESHOLD['texture_kurtosis'] else 0)
        confidence += (1 if solidity < self.IRREGULAR_THRESHOLD['solidity'] else 0)

        print(f'Irregular confidence = {confidence} / {len(self.IRREGULAR_THRESHOLD.keys())} ({(confidence / len(self.IRREGULAR_THRESHOLD.keys())) * 100}%)')
        irregular_confidence = (confidence / len(self.IRREGULAR_THRESHOLD.keys())) * 100

        return irregular_confidence

    def detect_morphology(self, image) -> (dict):
        """
        Detect the morphology of the given galaxy image and return confidence scores for each galaxy type.

        :param image: The image of the galaxy.
        :return: Dictionary containing confidence scores for each galaxy type.
        """
        #return self._get_elliptical_confidence(image) # testing
    
        self.class_confidence['elliptical'] = self._get_elliptical_confidence(image)
        self.class_confidence['spiral'] = self._get_spiral_confidence(image)
        self.class_confidence['irregular'] = self._get_irregular_confidence(image)

        return self.class_confidence

