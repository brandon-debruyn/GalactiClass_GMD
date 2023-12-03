from common_imports import *

class GalacticClass_Helpers(object):
    """
    Helper class containing various static methods for galaxy image analysis.
    """

    def __init__(self) -> None:
        pass

    @staticmethod
    def get_largest_contour(contours):
        """
        Get the largest contour from a list of contours.

        :param contours: List of contours.
        :return: The largest contour based on area, or None if no contours are found.
        """
        return max(contours, key=cv2.contourArea) if contours else None

    @staticmethod
    def get_galaxy_ellipticity(image, galaxy_contour) -> tuple:
        """
        Calculate the ellipticity of a galaxy based on its nucleus contour.

        :param image: Galaxy image.
        :param galaxy_contour: Contour of the galaxy nucleus.
        :return: Centroid coordinates and the calculated hubble ellipticity of the galaxy.
        """
        centroid_x, centroid_y, hubble_ellipticity = None, None, None
        if galaxy_contour is not None and len(galaxy_contour) >= 5:
            ellipse = cv2.fitEllipse(galaxy_contour)
            cv2.ellipse(image, ellipse, (255, 0, 0), 2)

            (centroid_x, centroid_y), (MA, ma), angle = ellipse
            a = MA / 2
            b = ma / 2
            a, b = max(a, b), min(a, b)
            ellipticity = 1 - (b / a)
            hubble_ellipticity = int(10 * ellipticity)
        
        return centroid_x, centroid_y, hubble_ellipticity

    @staticmethod
    def elliptical_brightness_law(rad, central_brightness, scale_length) -> float:
        """
        Calculate the brightness at a given radius based on the elliptical monotonical brightness law.

        :param rad: Radius.
        :param central_brightness: Central brightness.
        :param scale_length: Scale length.
        :return: Calculated brightness at the given radius.
        """
        return central_brightness * (rad / scale_length + 1) ** (-2)

    @staticmethod
    def fit_brightness_profile(brightness_profile):
        """
        Fit a brightness profile according to the elliptical brightness law.

        :param brightness_profile: Brightness profile of an galaxy.
        :return: Fitted parameters or None if fitting fails.
        """
        radii = np.arange(len(brightness_profile))
        try:
            params, _ = curve_fit(GalacticClass_Helpers.elliptical_brightness_law, radii, brightness_profile, maxfev=10000)
            return params
        except RuntimeError:
            return None

    @staticmethod
    def calculate_r_squared(brightness_profile, fitted_params) -> float:
        """
        Calculate the R-squared value for the fit of the brightness profile.

        :param brightness_profile: Brightness profile of a galaxy.
        :param fitted_params: Fitted parameters from the brightness to brightness law fitting.
        :return: R-squared value indicating the goodness of fit.
        """
        radii = np.arange(len(brightness_profile))
        fitted_curve = GalacticClass_Helpers.elliptical_brightness_law(radii, *fitted_params)
        residuals = brightness_profile - fitted_curve
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((brightness_profile - np.mean(brightness_profile))**2)
        r_squared = 1 - (ss_res / ss_tot)
        
        return r_squared
    
    @staticmethod
    def calculate_color_brightness_profile(image, centroid_x=None, centroid_y=None):
        """
        Calculate the brightness profile of each color channel in a galaxy image.
        
        S. S. McGaugh, G. D. Bothun, and J. M. Schombert, “Galaxy selection and the Surface Brightness Distribution,” 
            The Astronomical Journal, vol. 110, p. 573, 1995. doi:10.1086/117543 

        :param image: Input galaxy image.
        :param centroid_x: x-coordinate of the galaxy's centroid (optional).
        :param centroid_y: y-coordinate of the galaxy's centroid (optional).
        :return: A dictionary containing the brightness profiles for the BGR color space.
        """
        # Determine galaxy center if not provided
        if centroid_x is None or centroid_y is None:
            centroid_x, centroid_y = image.shape[1] // 2, image.shape[0] // 2
        else:
            centroid_x, centroid_y = int(centroid_x), int(centroid_y)

        height, width = image.shape[:2]
        max_radius = np.sqrt((height/2)**2 + (width/2)**2)
        
        # Initializing color brightness profiles
        color_brightness_profiles = {'b': [], 'g': [], 'r': []}
        # Iterating over radii to calculate average brightness for each color channel
        for r in range(int(max_radius - (0.20 * max_radius))):
            mask = np.zeros((height, width), dtype=np.uint8)
            cv2.circle(mask, (centroid_x, centroid_y), r, 255, -1)
            for i, col in enumerate(['b', 'g', 'r']):
                masked_img = cv2.bitwise_and(image[:, :, i], image[:, :, i], mask=mask)
                total_brightness = np.sum(masked_img)
                area = np.pi * r**2
                average_brightness = total_brightness / area if area > 0 else 0
                color_brightness_profiles[col].append(average_brightness)
                
        return color_brightness_profiles
    
    @staticmethod
    def elliptical_brightness_profile(brightness_profile) -> bool:
        """
        Determine if the brightness profile is decreasing monotonically outward towards the edge
        of the galaxy. Some tolerance error is given for image noise.

        S. S. McGaugh, G. D. Bothun, and J. M. Schombert, “Galaxy selection and the Surface Brightness Distribution,” 
        The Astronomical Journal, vol. 110, p. 573, 1995. doi:10.1086/117543 

        :param brightness_profile: Brightness profile to be evaluated.
        :return: Boolean indicating if the brightness profile is decreasing monotonically outward towards the edge of the galaxy.
        """
        epsilon = int(0.10 * len(brightness_profile))  # 10% tolerance
        delta_e = 0  # Counter for deviations from expected profile

        # Checking for deviation from expected monotonic decrease
        for i in range(len(brightness_profile) - 1):
            if brightness_profile[i] < brightness_profile[i + 1]:
                delta_e += 1
            if delta_e > epsilon:
                return False

        return True
    
    @staticmethod
    def calculate_gini_coefficient(image) -> float:
        """
        Calculate the Gini coefficient for an image, a measure of inequality in pixel values.

        :param image: Input image.
        :return: Gini coefficient of the image.
        """
        # Subtracting the background level from the image
        background_level = np.median(image)
        image_subtracted = image - background_level
        image_subtracted[image_subtracted < 0] = 0

        # Sorting pixel values to calculate Gini coefficient
        flattened = image_subtracted.flatten()
        sorted_pixels = np.sort(flattened)
        n = len(sorted_pixels)
        index = np.arange(1, n+1)

        return (np.sum((2 * index - n - 1) * sorted_pixels)) / (n * np.sum(sorted_pixels)) if np.sum(sorted_pixels) != 0 else 0.0
    
    @staticmethod
    def calculate_asymmetry_index_cont(image) -> float:
        """
        Calculate the asymmetry index of a galaxy, indicating deviation from symmetry.

        C. J. Conselice, M. A. Bershady, and A. Jangren, “The Asymmetry of Galaxies: Physical Morphology for Nearby and High‐Redshift Galaxies,” 
        The Astrophysical Journal, vol. 529, no. 2, pp. 886–910, 2000. doi:10.1086/308300

        :param image: Input galaxy image.
        :return: Asymmetry index of the galaxy.
        """
        # Converting to grayscale if necessary
        if len(image.shape) > 2:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply gaussian blur to image using kernel to reduce noise and small features like stars
        # Mather, P. M. 2004. Computer Processing of Remotely Sensed Images, An Introduction. West Sussex. John Wiley & Sons Ltd
        gaus_blur_kern_5 = np.array([
            [1, 4, 6, 4, 1], 
            [4, 16, 24, 16, 4],
            [6, 24, 36, 24, 6], 
            [4, 16, 24, 16, 4], 
            [1, 4, 6, 4, 1]])
        img_gaus_blurred = cv2.filter2D(image, ddepth=-1, kernel=gaus_blur_kern_5 / (np.sum(gaus_blur_kern_5)))
        
        # Creating a binary mask based on a threshold and finding contours in the mask
        _, binary_mask = cv2.threshold(img_gaus_blurred, np.percentile(image, 95), 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # If no contours are found, return nothing
        if not contours:
            return binary_mask, None

        # Find the largest contour - this is the galaxy and fill the largest contour for the mask
        largest_contour = max(contours, key=cv2.contourArea)
        mask_largest = np.zeros_like(image)
        cv2.drawContours(mask_largest, [largest_contour], -1, color=255, thickness=cv2.FILLED)

        # Apply the mask to the original grayscale image
        masked_image = cv2.bitwise_and(image, image, mask=mask_largest)

        # Rotate the image by 180 degrees and Calculate the asymmetry index
        rotated = cv2.rotate(masked_image, cv2.ROTATE_180)        
        asymmetry_index = np.sum(np.abs(masked_image - rotated)) / np.sum(masked_image) if np.sum(masked_image) != 0 else 0
        
        return asymmetry_index

    @staticmethod
    def calculate_color_gradients(image):
        """
        Calculate the color gradients within a galaxy image.
        
        M. Franx and G. Illingworth, “Color gradients in elliptical galaxies,” 
        The Astrophysical Journal, vol. 359, 1990. doi:10.1086/185791

        :param image: Input galaxy image.
        :return: List of average gradients for each color channel.
        """
        if len(image.shape) != 3:
            raise ValueError("Color gradients require a color image")

        # Generating a binary mask
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Creating a binary mask based on a threshold and finding contours in the mask
        _, binary_mask = cv2.threshold(gray, np.percentile(gray, 90), 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return [0, 0, 0]

        # Find the largest contour - this is the galaxy and fill the largest contour for the mask
        largest_contour = max(contours, key=cv2.contourArea)
        mask_largest = np.zeros_like(gray)
        cv2.drawContours(mask_largest, [largest_contour], -1, color=255, thickness=cv2.FILLED)

        # loop over each color channel in the BGR image
        gradients = []
        for i in range(3):  # Assuming BGR format
            channel = image[:, :, i]
            
            # for each color channel a mask is generated to extract the current color channel intensities
            masked_channel = cv2.bitwise_and(channel, channel, mask=mask_largest)
            
            # the change in color intensity gets calculated with respect to x and y 
            grad_x = cv2.Sobel(masked_channel, cv2.CV_64F, 1, 0, ksize=5)
            grad_y = cv2.Sobel(masked_channel, cv2.CV_64F, 0, 1, ksize=5)
            
            # the gradient is calculated by considering the magnitude of the vectors and the mean is appended as the color channels gradient
            grad = np.sqrt(grad_x**2 + grad_y**2)
            gradients.append(np.mean(grad))

        return gradients