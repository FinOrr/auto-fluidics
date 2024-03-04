import cv2
import sys
import particle_detector as pdt

# Create an instance of the image processor
proc = pdt.ParticleImageProcessor()

# Create an output window for viewing data
win_name = 'ESP Stream'
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

# Escape key exits the stream
while cv2.waitKey(1) != 27:

    # Process live video from a networked camera
    proc.process_stream(stream_ip='192.168.1.34')

    # Display the processed images in the window
    if proc.image['processed'] is not None:
        cv2.imshow(win_name, proc.image['processed'])

    # If no processing performed, just display the source image
    else:
        cv2.imshow(win_name, proc.image['sample'])
    
# Clean up and exit
cv2.destroyWindow(win_name)
