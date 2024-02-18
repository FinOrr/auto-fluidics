import cv2
import os
import numpy as np
import time

class ParticleImageProcessor:
    def __init__(self):
        """
        Initialise the ParticleImageProcessor class.
        """
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
            'particle_centres' : [],                      # Locations of the particle centres
            'particle_radii' : [],                        # The radius of each particle
            'particle_radii_sorted' : [],                 # An asc list of the particle radii
            'particle_size_distribution': None,             # Particle size distribution
            'mean_particle_size': None,                     # Mean detected particle size
            'median_particle_size' : None,                  # Median detected particle size
            'std_dev_particle_size': None,                  # Std Dev of particle size
            'particle_density': None,                       # Particle density
            'particle_coverage': None,                      # Particle coverage
            'particle_circularity_distribution': None,      # Particle circularity distribution
            'mean_particle_circularity': None,              # Mean particle circularity
        }

        # Processing metrics
        self.processing_metrics = {
            'start_time' : None,        # Time of processing start
            'stop_time' : None,         # Time of processing stop
            'processing_time' : None,   # Amount of time taken to process one image           
            'processing_time_normalised' : None # Normalise for image size (seconds per pixel)        
        }

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

    def _load_image(self, filepath):
        """
        Load an image from the specified path and create a grayscale copy.

        Args:
            image_path (str): The path to the image file.
        """
        self.image['sample'] = cv2.imread(filepath)
        self.image['grayscale'] = cv2.cvtColor(self.image['sample'], cv2.COLOR_BGR2GRAY)
        self._get_image_meta_data(filepath)

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

    def _preprocess_low_quality(self):
        """
        Perform preprocessing for low-quality image.
        """
        # Define the motion kernel for deblurring (assuming motion in the x-axis)
        kernel_size = 9  # Adjust the kernel size according to the blur extent
        motion_kernel = np.zeros((kernel_size, kernel_size))
        motion_kernel[int((kernel_size - 1)/2), :] = np.ones(kernel_size)
        motion_kernel /= kernel_size

        # Perform Wiener deconvolution for motion deblurring
        self.image['preprocessed'] = cv2.filter2D(self.image['grayscale'], -1, motion_kernel)

        # Equalise and threshold
        self.image['preprocessed'] = cv2.equalizeHist(self.image['preprocessed'])
        _, self.image['preprocessed'] = cv2.threshold(self.image['preprocessed'], 24, 255, cv2.THRESH_BINARY)

        # Apply Sobel edge detection
        sobel_x = cv2.Sobel(self.image['preprocessed'], cv2.CV_64F, 1, 0, ksize=5)
        sobel_y = cv2.Sobel(self.image['preprocessed'], cv2.CV_64F, 0, 1, ksize=5)

        # Combine the gradient image to obtain the magnitude
        sobel_magnitude = cv2.magnitude(sobel_x, sobel_y)

        # Normalise the Sobel edge magnitude image
        sobel_magnitude_normalized = cv2.normalize(sobel_magnitude, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

        # Threshold the Sobel edge magnitude image to obtain a binary edge image
        _, self.image['preprocessed'] = cv2.threshold(sobel_magnitude_normalized, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    def _preprocess_high_quality(self):
        """
        Perform preprocessing on a high-quality image.
        """
        # Apply Gaussian blurring
        self.image['preprocessed'] = cv2.fastNlMeansDenoising(self.image['grayscale'], None, h=10, templateWindowSize=30, searchWindowSize=30)
        self.image['preprocessed'] = cv2.GaussianBlur(self.image['preprocessed'], (3, 3), 16)

        # Define the kernel for dilation
        kernel = np.ones((3,3), np.uint8)
        # Perform dilation operation
        self.image['preprocessed'] = cv2.dilate(self.image['preprocessed'], kernel, iterations=1)

    def _preprocess_image(self):
        if (self.image_meta['blurriness'] < 10):
            self._preprocess_low_quality()
        else:
            self._preprocess_high_quality()

    def _get_image_meta_data(self, filepath):
        self.image_meta['filepath'] = filepath  # Location of the image
        self.image_meta['filename'] = os.path.basename(filepath)
        
        # Store image dimension info
        self.image_meta['size_x'] = self.image['sample'].shape[1]  # Width of the image
        self.image_meta['size_y'] = self.image['sample'].shape[0]  # Height of the image

        # Calculate Illumination Uniformity
        illumination_uniformity = np.std(self.image['grayscale'])  # Standard deviation of pixel intensities

        # Calculate the noise levels using median
        self._measure_noise()

        # Measure the blurriness of the image
        self._measure_blurriness()

    def process_image(self, filepath=None):
        """
        Perform relevant image processing.
        """

        try:
            self._start_timer()
            if filepath is not None:
                try:
                    self.__init__()
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
        """

        contours, _ = cv2.findContours(self.image['preprocessed'], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if (self._is_valid_particle(contour)):
                (x, y), radius = cv2.minEnclosingCircle(contour)
                centre = (int(x), int(y))
                radius = int(radius)
                self.particle_metrics['particle_centres'].append(centre)
                self.particle_metrics['particle_radii'].append(radius)

    def _detect_particles_high_quality(self):
        """
        Detect particles in the input image using Hough Circle Transform for high-quality images.
        """

        self.image['preprocessed'] = cv2.fastNlMeansDenoising(self.image['grayscale'], None, h=10, templateWindowSize=30, searchWindowSize=30)
        kernel = np.ones((5,5), np.uint8)
        self.image['preprocessed'] = cv2.dilate(self.image['preprocessed'], kernel, iterations=1)
        circles = cv2.HoughCircles(self.image['preprocessed'], cv2.HOUGH_GRADIENT, dp=1, minDist=40,
                                    param1=70, param2=50, minRadius=10, maxRadius=500)
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for circle in circles[0, :]:
                centre = (circle[0], circle[1])
                radius = circle[2]
                self.particle_metrics['particle_centres'].append(centre)
                self.particle_metrics['particle_radii'].append(radius)

    def _filter_particles(self):
        """
        Remove outlier particles and false detections.
        """
        # Convert particle centres to numpy array for processing
        particle_centres = np.array(self.particle_metrics['particle_centres'])
        particle_radii = np.array(self.particle_metrics['particle_radii'])

        # Remove outliers
        filtered_particle_centres, filtered_particle_radii = self._remove_outliers(particle_centres, particle_radii)

        # Update the class variables with filtered lists to remove outlier particles
        self.particle_metrics['particle_centres'] = filtered_particle_centres.tolist()
        self.particle_metrics['particle_radii'] = filtered_particle_radii.tolist()

    def _remove_outliers(self, centres, radii):
        """
        Remove outliers from particle centres and radii.
        """
        # Calculate the median and standard deviation of radii
        radius_median = np.median(radii)
        radius_std = np.std(radii)

        # Calculate the lower and upper bounds for radii
        lower_bound = radius_median - (1.5 * radius_std)
        upper_bound = radius_median + (1.5 * radius_std)

        # Mask for outliers
        mask = np.logical_and(radii >= lower_bound, radii <= upper_bound)

        # Filter particle centres and radii based on the mask
        filtered_centres = centres[mask]
        filtered_radii = radii[mask]

        return filtered_centres, filtered_radii
    
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
        IMAGE_QUALITY_THRESHOLD = 10
        if self.image_meta['blurriness'] < IMAGE_QUALITY_THRESHOLD:
            self._detect_particles_low_quality()
        else:
            self._detect_particles_high_quality()

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
        # Calculate circularity for each particle (circumference^2 / (4 * pi * area))
        self.particle_metrics['particle_circularity_distribution'] = [4 * np.pi * radius / (2 * np.pi * radius)**2 for radius in self.particle_metrics['particle_radii']]

        # Mean Particle Circularity
        self.particle_metrics['mean_particle_circularity'] = np.mean(self.particle_metrics['particle_circularity_distribution'])

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
proc = ParticleImageProcessor()
sample_image_folder = "/home/fin/projects/auto-fluidics/img/"
avg_time = process_all_images(proc, sample_image_folder)
print(f"Average processing time: {avg_time:.3f} s")