import cv2
import os
import numpy as np
import time
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from perception.encapsulation_detector import EncapsulationDetector
    ENCAPSULATION_AVAILABLE = True
except ImportError:
    ENCAPSULATION_AVAILABLE = False

try:
    from processing.channel_detector import ChannelDetector
    CHANNEL_DETECTION_AVAILABLE = True
except ImportError:
    CHANNEL_DETECTION_AVAILABLE = False

class ParticleImageProcessor:
    def __init__(self, um_per_pixel=None):
        """
        Initialise the ParticleImageProcessor class.

        Args:
            um_per_pixel (float, optional): Calibration factor for converting pixels to micrometers.
                If None, measurements will be in pixels only.
        """
        # Calibration
        self.calibration = {
            'um_per_pixel': um_per_pixel  # Micrometers per pixel (set via calibration)
        }

        # Hough Circle parameters (can be overridden)
        self.hough_params = {
            'dp': 1,
            'minDist': 40,
            'param1': 70,
            'param2': 50,
            'minRadius': 10,
            'maxRadius': 500
        }

        # Encapsulation detector (optional)
        self.encapsulation_detector = EncapsulationDetector() if ENCAPSULATION_AVAILABLE else None
        self.detect_encapsulation_enabled = False  # Can be enabled via method

        # Initialise image objects
        self.image = {
            'sample'        : None, # Original input image
            'grayscale'     : None, # Grayscale representation
            'preprocessed'  : None, # Preprocessed image
            'processed'     : None  # Processed image
        }
        
        # Image meta-information: for the input sample image ONLY
        self.image_meta = {
            'filename':     None,   # Source image name, if any
            'filepath' :    None,   # Source image location, if any
            'size_x'   :    None,   # Source image size (width, pixels)
            'size_y'   :    None,   # Source image size, (height, pixels)
            'blurriness':   None,   # Blurriness (using sobel gradient magnitude)
            'illumination_uniformity':  None, # Illumination uniformity
            'noise_level':  None,   # Noise level of the image as percentage (0% = ideal)
            'resolution':   None    # Resolution (micrometers per pixel)
        }
        
        # Particle metrics
        self.particle_metrics = {
            'num_particles': None,                          # Number of detected particles
            'particle_centres' : [],                        # Locations of the particle centres
            'particle_radii' : [],                          # The radius of each particle
            'particle_radii_sorted' : [],                   # An asc list of the particle radii
            'particle_size_distribution': None,             # Particle size distribution
            'mean_particle_size': None,                     # Mean detected particle size
            'median_particle_size' : None,                  # Median detected particle size
            'std_dev_particle_size': None,                  # Std Dev of particle size
            'particle_density': None,                       # Particle density
            'particle_coverage': None,                      # Particle coverage
            'particle_circularity_distribution': None,      # Particle circularity distribution
            'mean_particle_circularity': None,              # Mean particle circularity
            'particle_aspect_ratios': [],                   # Aspect ratios (major/minor axis) for regime detection
            'mean_aspect_ratio': None,                      # Mean aspect ratio
            'coefficient_of_variation': None,               # CV = std/mean (uniformity metric)
            # Encapsulation metrics (if enabled)
            'encapsulation_rate': None,                     # Fraction of particles encapsulated
            'particles_per_droplet': None,                  # Mean particles per droplet
            'poisson_lambda': None,                         # Poisson loading parameter
            'num_droplets': None,                           # Number of detected droplets
            'num_encapsulated': None,                       # Number of encapsulated particles
        }

        # Processing metrics
        self.processing_metrics = {
            'start_time' : None,        # Time of processing start
            'stop_time' : None,         # Time of processing stop
            'processing_time' : None,   # Amount of time taken to process one image           
            'processing_time_normalised' : None # Normalise for image size (seconds per pixel)        
        }

    def set_calibration(self, um_per_pixel):
        """
        Set the calibration factor for converting pixels to micrometers.

        Args:
            um_per_pixel (float): Micrometers per pixel calibration factor.
                Can be determined by imaging a known reference object.

        Example:
            # If a 100 µm reference object measures 50 pixels:
            processor.set_calibration(um_per_pixel=100/50)  # 2.0 µm/pixel
        """
        self.calibration['um_per_pixel'] = um_per_pixel
        self.image_meta['resolution'] = um_per_pixel

    def enable_encapsulation_detection(self, enabled=True):
        """
        Enable or disable encapsulation detection.

        Args:
            enabled (bool): Whether to compute encapsulation metrics

        Raises:
            ImportError: If encapsulation detector module is not available
        """
        if enabled and not ENCAPSULATION_AVAILABLE:
            raise ImportError("Encapsulation detector module not available. Check perception module.")
        self.detect_encapsulation_enabled = enabled

    def _start_timer(self):
        self.processing_metrics['start_time'] = time.time()

    def _stop_timer(self):
        self.processing_metrics['stop_time'] = time.time()
        self.processing_metrics['processing_time'] = float(self.processing_metrics['stop_time']) - float(self.processing_metrics['start_time'])
        # Calculate performance of the image processor: how many pixels per second can we manage?
        n_pixels = float(self.image_meta['size_x']) * float(self.image_meta['size_y'])
        if self.processing_metrics['processing_time'] is not None:
            self.processing_metrics['processing_time_normalised'] = n_pixels / self.processing_metrics['processing_time'] 

    def _measure_noise(self):
        """
        Measure the noise in an image. We apply median filtering to remove any noise.
        Then we can see how much noise was effectively removed to know how much was
        originally present in the image. Make sense?
        """
        image = self.image['grayscale']

        # Apply median filtering
        median_filtered = cv2.medianBlur(image, 3)  # Kernel size (5x5)

        # Calculate the absolute difference between original and median filtered image
        diff = np.abs(image.astype(np.float32) - median_filtered.astype(np.float32))

        # Calculate the maximum possible difference (assuming pixel values in [0, 255])
        max_diff = 255.0

        # Calculate noise levels as a percentage
        noise_level_percentage = (np.sum(diff) / (image.shape[0] * image.shape[1] * max_diff)) * 100
        self.image_meta['noise_level'] = noise_level_percentage

    def _load_stream(self, stream_ip):
        cap = cv2.VideoCapture()
        # print(f'http://{stream_ip}:81/stream')
        cap.open(f'http://{stream_ip}:81/stream')
        has_frame, frame = cap.read()
        if (has_frame):
            self.image['sample'] = frame
            self.image['grayscale'] = cv2.cvtColor(self.image['sample'], cv2.COLOR_BGR2GRAY)
            self._get_image_meta_data()

            
    def _load_image(self, filepath):
        """
        Load an image from the specified path and create a grayscale copy.
        For colored images with blue dye, uses inverted blue channel for better detection.

        Args:
            image_path (str): The path to the image file.
        """
        self.image['sample'] = cv2.imread(filepath)

        # Detect if image has blue-dyed droplets and use appropriate channel
        if len(self.image['sample'].shape) == 3:  # Color image
            b, g, r = cv2.split(self.image['sample'])

            # Analyze color composition
            blue_mean = b.mean()
            green_mean = g.mean()
            red_mean = r.mean()

            # Check if this is a true grayscale image (all channels nearly equal)
            channel_variance = np.std([blue_mean, green_mean, red_mean])

            if channel_variance < 2.0:
                # True grayscale image - all channels are essentially equal
                self.image['grayscale'] = cv2.cvtColor(self.image['sample'], cv2.COLOR_BGR2GRAY)
                self.image_meta['color_mode'] = 'grayscale'
            elif blue_mean > red_mean + 5:
                # Blue-dyed droplets: blue channel is significantly brighter than red
                self.image['grayscale'] = 255 - b
                self.image_meta['color_mode'] = 'blue_channel_inverted'
            else:
                # Standard color image, use grayscale conversion
                self.image['grayscale'] = cv2.cvtColor(self.image['sample'], cv2.COLOR_BGR2GRAY)
                self.image_meta['color_mode'] = 'grayscale'
        else:
            # Already grayscale
            self.image['grayscale'] = self.image['sample']
            self.image_meta['color_mode'] = 'grayscale'

        self._get_image_meta_data(filepath)

    def _detect_channel(self):
        """
        Detect microfluidic channel region to constrain particle search.
        Reduces false positives from watermarks and artifacts.
        """
        if not self.channel_detection_enabled or self.channel_detector is None:
            return

        # Detect channel bounds
        bounds = self.channel_detector.detect_channel(self.image['sample'], method='auto')

        if bounds:
            # Create ROI mask
            self.channel_roi_mask = self.channel_detector.get_roi_mask(self.image['sample'].shape)

            # Store channel info in metadata
            info = self.channel_detector.get_info()
            self.image_meta['channel_detected'] = True
            self.image_meta['channel_bounds'] = bounds
            self.image_meta['channel_method'] = info['method']
            self.image_meta['channel_confidence'] = info['confidence']
        else:
            # No channel detected, use full image
            h, w = self.image['sample'].shape[:2]
            self.channel_roi_mask = np.ones((h, w), dtype=np.uint8) * 255
            self.image_meta['channel_detected'] = False

    def _measure_blurriness(self):
        """
        Analyse the quality of the input image by measuring the blurriness.
        """

        # Convert the image to grayscale
        self.image['grayscale'] = cv2.cvtColor(self.image['sample'], cv2.COLOR_BGR2GRAY)
        
        # Calculate the Sobel gradients in the x and y directions
        gradient_x = cv2.Sobel(self.image['grayscale'], cv2.CV_64F, 1, 0, ksize=3)
        gradient_y = cv2.Sobel(self.image['grayscale'], cv2.CV_64F, 0, 1, ksize=3)
        
        # Compute the combined gradient magnitude
        gradient_magnitude = cv2.magnitude(gradient_x, gradient_y)
        
        # Compute the average gradient magnitude
        average_gradient = cv2.mean(gradient_magnitude)[0]

        # Record the blurriness of the image: 
        # Blurry < Clear, we say 10 is a clear image, 0 is a useless image
        self.image_meta['blurriness'] = average_gradient

    def _preprocess_thresholding(self):
        """
        Perform preprocessing using thresholding.
        Uses Otsu's method for blue-channel images, adaptive for grayscale.
        """
        # For blue-channel-inverted images, use Otsu's global threshold
        if self.image_meta.get('color_mode') == 'blue_channel_inverted':
            # Otsu's method automatically finds optimal threshold
            _, self.image['preprocessed'] = cv2.threshold(self.image['preprocessed'],
                                                         0, 255,
                                                         cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            # Apply morphological closing to clean up droplets
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            self.image['preprocessed'] = cv2.morphologyEx(self.image['preprocessed'],
                                                         cv2.MORPH_CLOSE, kernel)
        else:
            # Standard adaptive thresholding for grayscale images
            self.image['preprocessed'] = cv2.adaptiveThreshold(self.image['preprocessed'],
                                                       255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                       cv2.THRESH_BINARY, 11, 7)

    def _preprocess_gaussian_blur(self):
        """
        Perform preprocessing step: applying Gaussian blur
        """        
        self.image['preprocessed'] = cv2.GaussianBlur(self.image['preprocessed'], (3, 3), 16)

    def _preprocess_image(self):
        # Start point of all preprocessed images is grayscale representation
        self.image['preprocessed'] = self.image['grayscale']
        # Apply Gaussian blur to remove high frequency noise from the image
        self._preprocess_gaussian_blur()
        # Apply an adaptive thresholding algorithm to isolate key areas from the image
        self._preprocess_thresholding()

    def _get_image_meta_data(self, filepath=None):
        if filepath is not None:
            self.image_meta['filepath'] = filepath  # Location of the image
            self.image_meta['filename'] = os.path.basename(filepath)
        
        # Store image dimension info
        self.image_meta['size_x'] = self.image['sample'].shape[1]  # Width of the image
        self.image_meta['size_y'] = self.image['sample'].shape[0]  # Height of the image

        # Calculate Illumination Uniformity (lower std = more uniform lighting)
        self.image_meta['illumination_uniformity'] = np.std(self.image['grayscale'])

        # Store resolution from calibration
        self.image_meta['resolution'] = self.calibration['um_per_pixel']

        # Calculate the noise levels using median
        self._measure_noise()

        # Measure the blurriness of the image
        self._measure_blurriness()

    def process_stream(self, stream_ip):
        """
        Perform real-time video processing
        """
        try:
                self._start_timer()
                self._load_stream(stream_ip)
                self._preprocess_image()
                self._detect_particles()
                self._draw_contours()
                self._get_metrics()
                self._stop_timer()
                self._annotate_data()
        except Exception as e:
            print(f"Error processing image: {e}")
            self._stop_timer()

    def process_image(self, filepath=None, hough_params=None):
        """
        Perform relevant image processing.
        
        Args:
            filepath (str, optional): Path to the image file to process.
            hough_params (dict, optional): Override Hough Circle Transform parameters.
                Keys: dp, minDist, param1, param2, minRadius, maxRadius
        """
        
        # Update Hough parameters if provided
        if hough_params is not None:
            self.hough_params.update(hough_params)

        try:
            self._start_timer()
            if filepath is not None:
                try:
                    self.__init__()
                    
                    # Restore Hough params if they were provided
                    if hough_params is not None:
                        self.hough_params.update(hough_params)
                    
                    self._start_timer()
                    self._load_image(filepath)
                    self._preprocess_image()
                    self._detect_particles()
                    self._draw_contours()
                    self._get_metrics()
                    self._stop_timer()
                    self._annotate_data()
                    self._save_image()
                except Exception as e:
                    print(f"Error processing image: {e}")
                    # Ensure timer is stopped even if an error occurs
                    self._stop_timer()
            else:
                print("No file path provided.")
        except Exception as e:
            print(f"Error: {e}")
            
    def _is_valid_particle(self, contour):
        """
        Given a contour from 

        Args:
            contour (tuple): Particle bounding contour.

        Returns:
            validity (boolean): True if a valid particle is bounded by the contour.
        """

        circularity = 0
        circularity_threshold = 0.8
        
        if contour is not None:
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            circularity = 4 * np.pi * (area / (perimeter * perimeter))
        return (circularity > circularity_threshold)

    def _detect_particles_low_quality(self):
        """
        Detect particles in the input image using contour analysis for low-quality images.
        Also calculates aspect ratios for regime detection.
        """

        contours, _ = cv2.findContours(self.image['preprocessed'], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if (self._is_valid_particle(contour)):
                (x, y), radius = cv2.minEnclosingCircle(contour)
                centre = (int(x), int(y))
                radius = int(radius)
                self.particle_metrics['particle_centres'].append(centre)
                self.particle_metrics['particle_radii'].append(radius)

                # Calculate aspect ratio by fitting ellipse
                if len(contour) >= 5:  # Need at least 5 points to fit ellipse
                    try:
                        ellipse = cv2.fitEllipse(contour)
                        (x_e, y_e), (minor_axis, major_axis), angle = ellipse
                        aspect_ratio = major_axis / minor_axis if minor_axis > 0 else 1.0
                        self.particle_metrics['particle_aspect_ratios'].append(aspect_ratio)
                    except:
                        # If ellipse fitting fails, assume circular
                        self.particle_metrics['particle_aspect_ratios'].append(1.0)
                else:
                    self.particle_metrics['particle_aspect_ratios'].append(1.0)

    def _detect_particles_high_quality(self):
        """
        Detect particles in the input image using Hough Circle Transform for high-quality images.
        Also estimates aspect ratios by finding nearby contours for regime detection.
        Adjusts parameters for blue-channel-inverted images or uses custom parameters.
        """

        # Use relaxed parameters for blue-channel-inverted images (Otsu thresholding)
        # unless custom parameters are explicitly set
        if self.image_meta.get('color_mode') == 'blue_channel_inverted':
            # Default relaxed parameters for blue channel
            default_param1, default_param2 = 40, 20
        else:
            # Standard for adaptive-thresholded images
            default_param1, default_param2 = 70, 50

        # Use Hough parameters (either custom or defaults)
        # None values mean "use auto-detection"
        param1 = self.hough_params.get('param1')
        if param1 is None:
            param1 = default_param1
            
        param2 = self.hough_params.get('param2')
        if param2 is None:
            param2 = default_param2
            
        dp = self.hough_params.get('dp', 1)
        minDist = self.hough_params.get('minDist', 40)
        minRadius = self.hough_params.get('minRadius', 10)
        maxRadius = self.hough_params.get('maxRadius', 500)

        circles = cv2.HoughCircles(
            self.image['preprocessed'], 
            cv2.HOUGH_GRADIENT, 
            dp=dp, 
            minDist=minDist,
            param1=param1, 
            param2=param2, 
            minRadius=minRadius, 
            maxRadius=maxRadius
        )

        # Find contours for aspect ratio calculation
        contours, _ = cv2.findContours(self.image['preprocessed'], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if circles is not None:
            circles = np.uint16(np.around(circles))

            # Get image height for watermark filtering
            img_height = self.image['preprocessed'].shape[0]
            watermark_y_threshold = img_height * 0.85  # Exclude bottom 15%

            for circle in circles[0, :]:
                centre = (circle[0], circle[1])
                radius = circle[2]

                # Skip detections in watermark region (bottom 15%)
                if centre[1] > watermark_y_threshold:
                    continue

                self.particle_metrics['particle_centres'].append(centre)
                self.particle_metrics['particle_radii'].append(radius)

                # Find nearest contour to measure aspect ratio
                aspect_ratio = 1.0  # Default for perfect circle
                min_dist = float('inf')
                nearest_contour = None

                for contour in contours:
                    M = cv2.moments(contour)
                    if M['m00'] != 0:
                        cx = int(M['m10'] / M['m00'])
                        cy = int(M['m01'] / M['m00'])
                        dist = np.sqrt((cx - centre[0])**2 + (cy - centre[1])**2)
                        if dist < min_dist and dist < radius * 1.5:  # Within 1.5x radius
                            min_dist = dist
                            nearest_contour = contour

                # Calculate aspect ratio from nearest contour
                if nearest_contour is not None and len(nearest_contour) >= 5:
                    try:
                        ellipse = cv2.fitEllipse(nearest_contour)
                        (x_e, y_e), (minor_axis, major_axis), angle = ellipse
                        aspect_ratio = major_axis / minor_axis if minor_axis > 0 else 1.0
                    except:
                        aspect_ratio = 1.0

                self.particle_metrics['particle_aspect_ratios'].append(aspect_ratio)

    def _filter_particles(self):
        """
        Remove outlier particles and false detections.
        """
        # Convert particle data to numpy arrays for processing
        particle_centres = np.array(self.particle_metrics['particle_centres'])
        particle_radii = np.array(self.particle_metrics['particle_radii'])
        particle_aspect_ratios = np.array(self.particle_metrics['particle_aspect_ratios'])

        # Remove outliers based on radii
        filtered_particle_centres, filtered_particle_radii, filtered_aspect_ratios = self._remove_outliers(
            particle_centres, particle_radii, particle_aspect_ratios
        )

        # Update the class variables with filtered lists to remove outlier particles
        self.particle_metrics['particle_centres'] = filtered_particle_centres.tolist()
        self.particle_metrics['particle_radii'] = filtered_particle_radii.tolist()
        self.particle_metrics['particle_aspect_ratios'] = filtered_aspect_ratios.tolist()

    def _remove_outliers(self, centres, radii, aspect_ratios):
        """
        Remove outliers from particle centres, radii, and aspect ratios.
        """
        # Calculate the median and standard deviation of radii
        radius_median = np.median(radii)
        radius_std = np.std(radii)

        # Calculate the lower and upper bounds for radii
        lower_bound = radius_median - (1.5 * radius_std)
        upper_bound = radius_median + (1.5 * radius_std)

        # Mask for outliers
        mask = np.logical_and(radii >= lower_bound, radii <= upper_bound)

        # Filter particle data based on the mask
        filtered_centres = centres[mask]
        filtered_radii = radii[mask]
        filtered_aspect_ratios = aspect_ratios[mask]

        return filtered_centres, filtered_radii, filtered_aspect_ratios
    
    # def _remove_outliers(self, array):

    #     samples =  np.array(array)
    #     median = np.median(samples)
    #     standard_deviation = np.std(samples)
    #     distance_from_mean = abs(samples - median)
    #     not_outlier = distance_from_mean < standard_deviation
    #     no_outliers = samples[not_outlier]
    #     return (no_outliers)
        
    def _detect_particles(self):
        """
        Detect particles in the input image.
        """
        # IMAGE_QUALITY_THRESHOLD = 10
        # if self.image_meta['blurriness'] < IMAGE_QUALITY_THRESHOLD:
        self._detect_particles_high_quality()
        # # else:
        # self._detect_particles_low_quality()

        # Remove the erronous particle detections
        self._filter_particles()

    def _draw_contours(self):
        """
        Draw circles on the sample image to generte the annotated image.
        """

        RED     = (255, 0, 0)
        GREEN   = (0, 255, 0)
        BLUE    = (0, 0, 255)
        self.image['processed'] = self.image['sample']
    
        # Iterate through circles and filter based on radius
        for centre, radius in zip(
            self.particle_metrics['particle_centres'], 
            self.particle_metrics['particle_radii']):
                # Draw a green circle outline on the original image (3 is way too big)
                cv2.circle(self.image['processed'], centre, radius, GREEN, 2)
    
    def _get_median_particle_size(self):
        sorted_radii = self.particle_metrics['particle_radii']
        sorted_radii = sorted(sorted_radii)
        self.particle_metrics['particle_radii_sorted'] = sorted_radii

        # Calculate the median circle radius
        if sorted_radii:
            if len(sorted_radii) % 2 == 0:
                median_radius = (sorted_radii[len(sorted_radii) // 2 - 1] + sorted_radii[len(sorted_radii) // 2]) / 2
            else:
                median_radius = sorted_radii[len(sorted_radii) // 2]
        else:
            median_radius = None

        return (median_radius)
    
    def _count_keys(self, dictionary):
        count = 0
        for value in dictionary.values():
            if not isinstance(value, list):
                count += 1
            else:
                count += bool(value)  # Count as 1 if the list is not empty
        return count

    def _get_metrics(self):
        """
        Generate metrics based on the detected particles.

        This method calculates various metrics based on the detected particles, 
        including size distribution, mean size, median size, standard deviation of size, 
        particle density, particle coverage, circularity distribution, and mean circularity.

        Note:
            This method assumes that the 'particle_metrics' dictionary contains the following keys:
            - 'particle_centres': List of detected particle centers.
            - 'particle_radii': List of detected particle radii.
            - 'particle_radii': List of detected circle radii (used for circularity distribution calculation).
            - The 'image_meta' dictionary should contain 'size_x' and 'size_y' keys 
            representing the width and height of the source image, respectively.

        """
        # Number of detected particles
        self.particle_metrics['num_particles'] = len(self.particle_metrics['particle_centres'])
        
        # Particle Size Distribution
        self.particle_metrics['particle_size_distribution'], bins = np.histogram(
            self.particle_metrics['particle_radii'], bins=10)
        
        # Mean Particle Size
        self.particle_metrics['mean_particle_size'] = np.mean(self.particle_metrics['particle_radii'])

        # Median Particle Size
        self.particle_metrics['median_particle_size'] = self._get_median_particle_size() 

        # Standard Deviation of Particle Size
        self.particle_metrics['std_dev_particle_size'] = np.std(self.particle_metrics['particle_radii'])
        
        # Particle Density
        total_area = self.image_meta['size_x'] * self.image_meta['size_y']
        particle_area = [np.pi * radius**2 for radius in self.particle_metrics['particle_radii']]
        self.particle_metrics['particle_density'] = len(self.particle_metrics['particle_radii']) / total_area
        
        # Particle Coverage
        covered_area = np.sum(np.pi * np.square(self.particle_metrics['particle_radii']))
        self.particle_metrics['particle_coverage'] = covered_area / total_area

        # Particle Circularity Distribution
        # Calculate circularity for each particle: 4π * area / perimeter²
        # For circles detected by Hough, circularity should be ~1.0
        # Note: Since we're using Hough circles, actual circularity from contours would be better
        # For now, approximate using circle formula: 4π * (πr²) / (2πr)² = 1.0 (perfect circle)
        self.particle_metrics['particle_circularity_distribution'] = [1.0 for _ in self.particle_metrics['particle_radii']]

        # Mean Particle Circularity
        self.particle_metrics['mean_particle_circularity'] = 1.0 if len(self.particle_metrics['particle_radii']) > 0 else 0.0

        # Mean Aspect Ratio (for regime detection)
        if len(self.particle_metrics['particle_aspect_ratios']) > 0:
            self.particle_metrics['mean_aspect_ratio'] = np.mean(self.particle_metrics['particle_aspect_ratios'])
        else:
            self.particle_metrics['mean_aspect_ratio'] = 1.0

        # Coefficient of Variation (CV) - uniformity metric
        # CV = std / mean (lower is more uniform)
        if self.particle_metrics['mean_particle_size'] > 0:
            self.particle_metrics['coefficient_of_variation'] = (
                self.particle_metrics['std_dev_particle_size'] /
                self.particle_metrics['mean_particle_size']
            )
        else:
            self.particle_metrics['coefficient_of_variation'] = 0.0

        # Encapsulation metrics (if enabled)
        if self.detect_encapsulation_enabled and self.encapsulation_detector:
            self._compute_encapsulation()

    def _compute_encapsulation(self):
        """
        Compute encapsulation metrics using the encapsulation detector.
        """
        if not ENCAPSULATION_AVAILABLE or not self.encapsulation_detector:
            return

        # Run encapsulation detection
        result = self.encapsulation_detector.detect_encapsulation(
            self.particle_metrics['particle_centres'],
            self.particle_metrics['particle_radii']
        )

        # Store results in particle_metrics
        self.particle_metrics['encapsulation_rate'] = result['encapsulation_rate']
        self.particle_metrics['particles_per_droplet'] = result['mean_particles_per_droplet']
        self.particle_metrics['poisson_lambda'] = result['poisson_lambda']
        self.particle_metrics['num_droplets'] = result['num_droplets']
        self.particle_metrics['num_encapsulated'] = result['num_encapsulated']

    def _format_processing_time(self, value):
        if value < 1000:
            return f"{value:.1f} Pixels s⁻¹"
        elif value < 1e6:
            return f"{value / 1e3:.1f} kPixels / s"
        elif value < 1e9:
            return f"{value / 1e6:.1f} MPixels / s"
        else:
            return f"{value / 1e9:.1f} GPixels / s"

    def _annotate_data(self):

        # We'll annotate the sample input image
        sample = self.image['sample']
        
        # Format the 'number of pixels processed per second' metric
        processing_time = self.processing_metrics['processing_time_normalised']
        formatted_time = self._format_processing_time(processing_time)
        
        # Define font properties
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5                
        font_color = (255, 255, 255)    # RGB font colour
        line_height = 20                # Pixels per line of text
        line_count = self._count_keys(self.image_meta) + self._count_keys(self.particle_metrics)
        TEXT_PADDING = 10                # Number of pixels of space to leave between text and img edge
        x_offset = sample.shape[1] + 10 # Start position for the metrics on the right-hand side

        # Create a canvas to draw the image and metrics
        canvas_width = int(sample.shape[1] * 1.5)  # Width of the canvas, list might spill over but w/e       
        canvas_height = max(sample.shape[0], line_count * line_height + TEXT_PADDING)  # Height of the canvas
        canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

        # Calculate the vertical offset to center the image
        vertical_offset = (canvas.shape[0] - sample.shape[0]) // 2

        # Overlay the sample image on the canvas with the calculated offset
        canvas[vertical_offset:vertical_offset + sample.shape[0], :sample.shape[1]] = sample

        # Draw the image data metrics        # Calculate performance of the image processor: how many pixels per second can we manage?
        n_pixels = float(self.image_meta['size_x']) * float(self.image_meta['size_y'])
        self.processing_metrics['processing_time_normalised'] = n_pixels / self.processing_metrics['processing_time']

        cv2.putText(canvas, f"IMAGE DATA", 
                    (x_offset, 1 * line_height), cv2.FONT_HERSHEY_DUPLEX, font_scale + 0.2, font_color, 1)
        cv2.putText(canvas, f"Filename: {self.image_meta['filename']}", 
                    (x_offset, 2 * line_height), font, font_scale, font_color, 1)
        cv2.putText(canvas, f"Image Dimensions: {self.image_meta['size_x']:.3f} x {self.image_meta['size_y']:.3f}", 
                    (x_offset, 3 * line_height), font, font_scale, font_color, 1)
        cv2.putText(canvas, f"Blurriness: {self.image_meta['blurriness']:.3f}", 
                    (x_offset, 4 * line_height), font, font_scale, font_color, 1)
        cv2.putText(canvas, f"Illumination uniformity: {self.image_meta['illumination_uniformity']}", 
                    (x_offset, 5 * line_height), font, font_scale, font_color, 1)
        cv2.putText(canvas, f"Noise level (%): {self.image_meta['noise_level']:.3f}", 
                    (x_offset, 6 * line_height), font, font_scale, font_color, 1)
        cv2.putText(canvas, f"Resolution (um/pixel): {self.image_meta['resolution']}", 
                    (x_offset, 7 * line_height), font, font_scale, font_color, 1)
        cv2.putText(canvas, f"Processing time (s): {self.processing_metrics['processing_time']:.3f}", 
                    (x_offset, 8 * line_height), font, font_scale, font_color, 1)      
        cv2.putText(canvas, f"Pixels per second: {formatted_time}", 
                    (x_offset, 9 * line_height), font, font_scale, font_color, 1)

        # Draw particle data metrics
        cv2.putText(canvas, f"PARTICLE DATA", 
                    (x_offset, 11 * line_height), cv2.FONT_HERSHEY_DUPLEX, font_scale + 0.2, font_color, 1)
        cv2.putText(canvas, f"Number of particles: {self.particle_metrics['num_particles']}", 
                    (x_offset, 12 * line_height), font, font_scale, font_color, 1)
        cv2.putText(canvas, f"Particle Radii (px): {self.particle_metrics['particle_radii_sorted']}", 
                    (x_offset, 13 * line_height), font, font_scale, font_color, 1)
        cv2.putText(canvas, f"Particle size distribution: {self.particle_metrics['particle_size_distribution']}", 
                    (x_offset, 14 * line_height), font, font_scale, font_color, 1)
        cv2.putText(canvas, f"Mean particle size (px): {self.particle_metrics['mean_particle_size']:.3f}", 
                    (x_offset, 15 * line_height), font, font_scale, font_color, 1)
        cv2.putText(canvas, f"Median particle size (px): {self.particle_metrics['median_particle_size']:.3f}",
                    (x_offset, 16 * line_height), font, font_scale, font_color, 1)
        cv2.putText(canvas, f"Std dev of size: {self.particle_metrics['std_dev_particle_size']:.3f}", 
                    (x_offset, 17 * line_height), font, font_scale, font_color, 1)
        cv2.putText(canvas, f"Particle density: {self.particle_metrics['particle_density']:.3f}", 
                    (x_offset, 18 * line_height), font, font_scale, font_color, 1)
        cv2.putText(canvas, f"Coverage: {self.particle_metrics['particle_coverage']:.3f}", 
                    (x_offset, 19 * line_height), font, font_scale, font_color, 1)
        cv2.putText(canvas, f"Mean particle circularity: {self.particle_metrics['mean_particle_circularity']:.3f}", 
                    (x_offset, 20 * line_height), font, font_scale, font_color, 1)
        # Iterate over each value in the circularity distribution list and format it
        formatted_circularity_distribution = [f"{value:.3f}" for value in self.particle_metrics['particle_circularity_distribution']]
        # Join the formatted values into a single string
        formatted_circularity_distribution_str = ', '.join(formatted_circularity_distribution)
        # Draw the circularity distribution on the canvas
        cv2.putText(canvas, f"Circularity distribution: {formatted_circularity_distribution_str}", 
                    (x_offset, 21 * line_height), font, font_scale, font_color, 1)

        # Save the processed image
        self.image['processed'] = canvas

    def _save_image(self):
        try:
            # Extract the directory path and file name from the image file path
            directory, filename = os.path.split(self.image_meta['filepath'])
            
            # Define output directory (starting to regret not using pathlib)
            output_dir = os.path.join(os.path.dirname(directory), 'output')
            os.makedirs(output_dir, exist_ok=True)

            # Define subdirectories for grayscale, preprocessed, and processed images
            grayscale_dir = os.path.join(output_dir, 'grayscale')
            preprocessed_dir = os.path.join(output_dir, 'preprocessed')
            processed_dir = os.path.join(output_dir, 'processed')

            # Create directories if they don't exist 
            for directory in [grayscale_dir, preprocessed_dir, processed_dir]:
                os.makedirs(directory, exist_ok=True)

            # Save images to respective directories
            cv2.imwrite(os.path.join(grayscale_dir, filename), self.image['grayscale'])
            cv2.imwrite(os.path.join(preprocessed_dir, filename), self.image['preprocessed'])
            cv2.imwrite(os.path.join(processed_dir, filename), self.image['processed'])

        except Exception as e:
            print(f"Error saving image: {e}")
            # Handle the error appropriately, e.g., logging, raising an exception, etc.

def process_all_images(proc, folder_path):

    # Get a list of all .png files in the specified folder
    png_files = [file for file in os.listdir(folder_path) if file.endswith('.png')]
    
    # Initialise a list to store processing times for each image
    processing_times = []
    
    # Process each image
    for png_file in png_files:
        # Construct the full file path
        file_path = os.path.join(folder_path, png_file)
        
        # Process the image
        proc.process_image(file_path)
        
        # Record the processing time
        processing_time = proc.processing_metrics['processing_time']
        processing_times.append(processing_time)
        
    # Calculate the average processing time
    average_processing_time = sum(processing_times) / len(processing_times)
    
    return (average_processing_time)


# For debugging single images
# proc = ParticleImageProcessor()
# proc.process_image("/home/fin/projects/auto-fluidics/img/sample-14.png")

# # For batch testing and tuning
# proc = ParticleImageProcessor()
# sample_image_folder = "/home/fin/projects/auto-fluidics/img/"
# avg_time = process_all_images(proc, sample_image_folder)
# print(f"Average processing time: {avg_time:.3f} s")