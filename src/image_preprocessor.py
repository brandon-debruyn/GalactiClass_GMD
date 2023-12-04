from common_imports import *

class GalactiClass_PreProcessing(object):

    @staticmethod
    def crop_to_galaxy(galaxy_mask, image):
        """
            Find and crop to largest contour found in a image using the mask of galaxy.

            :param galaxy_mask: The mask of the galaxy.
            :param image: The image of the galaxy.
            :return: cropped image of galaxy (np.array).
        """
        contours, _ = cv2.findContours(galaxy_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        cropped_image = image[y:y+h, x:x+w]

        return cropped_image
    