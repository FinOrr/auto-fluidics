#!/usr/bin/env python3
"""
Live stream demo: reads from a networked ESP camera and displays
annotated droplet detections in a window.

Usage:
    python stream_demo.py
    python stream_demo.py --ip 192.168.1.34
"""

import argparse
import cv2
from bacha_vision.detection.droplet_detector import ParticleImageProcessor


def main():
    parser = argparse.ArgumentParser(description="Live stream droplet detection demo")
    parser.add_argument("--ip", default="192.168.1.34", help="Camera stream IP address")
    args = parser.parse_args()

    proc = ParticleImageProcessor()
    win_name = "Bacha Vision — Live Stream"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    print(f"Connecting to stream at {args.ip} — press Esc to exit")

    while cv2.waitKey(1) != 27:
        proc.process_stream(stream_ip=args.ip)

        if proc.image["processed"] is not None:
            cv2.imshow(win_name, proc.image["processed"])
        elif proc.image["sample"] is not None:
            cv2.imshow(win_name, proc.image["sample"])

    cv2.destroyWindow(win_name)


if __name__ == "__main__":
    main()
