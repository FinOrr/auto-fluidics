# Auto-Fluidics

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![GitHub top language](https://img.shields.io/github/languages/top/FinOrr/auto-fluidics.svg)](https://github.com/FinOrr/auto-fluidics)
[![GitHub issues](https://img.shields.io/github/issues/FinOrr/auto-fluidics.svg)](https://github.com/FinOrr/auto-fluidics/issues)
[![GitHub pull requests](https://img.shields.io/github/issues-pr/FinOrr/auto-fluidics.svg)](https://github.com/FinOrr/auto-fluidics/pulls)

This repository serves as a comprehensive exploration into the development of an affordable and open-source microfluidics camera equipped with advanced computer vision capabilities. The primary objective is to create a versatile solution capable of real-time particle analysis, alongside a robust control system for analysing particle metrics. Additionally, the project aims to provide a generic API for seamless interfacing with various pumps, including pressure, syringe, or peristaltic pumps.

## Key Features:

* Real-time particle analysis leveraging computer vision techniques
* Integration with a control system for efficient analysis of particle metrics
* Generic API facilitating seamless interaction with different types of pumps

## Motivation:

The motivation behind this project stems from the need for accessible tools in microfluidics research and experimentation. By combining affordability with open-source principles, I aim to democratise access to advanced microfluidics instrumentation, fostering innovation and collaboration within the scientific community.

The goal is to create a:
* solution that can perform particle analysis in real-time;
* control system to analyse the particle metrics (see [my PID autotuners](https://github.com/FinOrr/pid-autotuner)); and
* generic API for interfacing with pressure, syringe, or peristaltic pumps. 

## Table of Contents

1. [About the Project](#about-the-project)
2. [Project Status](#project-status)
3. [Getting Started](#getting-started)
    1. [Requirements](#requirements)
    2. [Getting the Source](#getting-the-source)
    3. [Building](#building)
    4. [Testing](#testing)
    5. [Example Usage](#example-usage)
    6. [Example Output](#example-output)
5. [Documentation](#documentation)
6. [Need Help?](#need-help)
7. [Contributing](#contributing)
8. [Further Reading](#further-reading)
9. [Authors](#authors)
10. [License](#license)
11. [Acknowledgments](#acknowledgements)

# About the Project

The motivation behind this project is to simplify the process of microfluidics experimentation. By providing an easy-to-use toolkit, we aim to empower researchers and enthusiasts to explore microfluidic phenomena without extensive technical barriers. 

This project seeks to democratise access to advanced microfluidics instrumentation, fostering collaboration and innovation in the field.

**Real-Time Particle Analysis:** 
    The project offers real-time analysis of particles within microfluidic systems, utilising computer vision algorithms.

**Control System Integration:**
    It seamlessly integrates with a control system for efficient analysis of particle metrics, enhancing experimental precision and protocol development.

**Versatile Pump Interface:**
    The toolkit includes a generic API for interfacing with pressure, syringe, or peristaltic pumps, ensuring compatibility with various experimental setups.

**User-Friendly Interface:** 
The project provides an intuitive and easy-to-use toolkit, enabling users to set up and conduct experiments with minimal technical expertise.

**[Back to top](#table-of-contents)**

# Project Status

This is an early work in progress, with what little free time I have.

Currently implemented using Python.

The methods of detection and preprocessing are constantly being updated, as more efficient pipelines are developed. Currently only tested using local images / videos, and video streams from an ESP32 device.

**[Back to top](#table-of-contents)**

## Getting Started

### Requirements

At a minimum you will need:

* [`git`](https://git-scm.com/downloads), version control and source code managment
* [`Python`](https://www.python.org/downloads/release/python-31012/), `3.10.12` is recommended
    - [opencv-python](https://pypi.org/project/opencv-python/), `4.9.0.80` library


**[Back to top](#table-of-contents)**



### Getting the Source

This project is hosted on GitHub. You can clone the project directly using this command:

```
git clone --recursive git@github.com:finorr/auto-fluidics.git
```

**[Back to top](#table-of-contents)**

### Example Usage

The system is designed to operate either in real-time processing mode, or as a post-processor. The system can detect and analyse particles in images, as well as videos.

```python
####
# Example script: process video using a Wi-Fi enabled microscope.
####
import cv2
import sys
import particle_detector as pdt

# Create an instance of the image processor
proc = pdt.ParticleImageProcessor()

# Create an output window for viewing data
win_name = 'Networked Stream'
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
```

### Example Output

![](output/processed/sample-12.png)

![](output/processed/sample-2.png)

![](output/processed/sample-17.png)

**[Back to top](#table-of-contents)**
### Building

The current design uses Python. When the system is expanded to support C++ (to enable GPU acceleration), then expect this section to be updated.

**[Back to top](#table-of-contents)**

### Testing

Unit tests incoming...

**[Back to top](#table-of-contents)**

## Documentation

Documentation incoming...

**[Back to top](#table-of-contents)**

## Need help?

If you need further assistance or have any questions, please file a GitHub issue or reach out on [Linkedin](https://www.linkedin.com/in/finorr/).

## Contributing

If you are interested in contributing to this project, please read our [contributing guidelines](docs/CONTRIBUTING.md).

## Authors

* **[Fin Orr](https://github.com/finorr)**

## License

See the [LICENSE](LICENSE) file for licensing details.

## Acknowledgments

Make any public acknowledgments here

**[Back to top](#table-of-contents)**

