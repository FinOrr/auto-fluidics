#!/usr/bin/env python3
"""
Interactive labeling tool for regime classification training data.

Helps you quickly label images as NO_FLOW, DRIPPING, JETTING, or CO_FLOW.

Usage:
    python label_training_data.py --input_dir img --output_dir training_data

Controls:
    1 - Label as NO_FLOW
    2 - Label as DRIPPING
    3 - Label as JETTING
    4 - Label as CO_FLOW
    s - Skip image
    q - Quit

Author: Auto-Fluidics Project
"""

import argparse
import cv2
import shutil
from pathlib import Path


def label_images(input_dir, output_dir):
    """Interactive labeling interface."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Create class directories
    for class_name in ['NO_FLOW', 'DRIPPING', 'JETTING', 'CO_FLOW']:
        (output_path / class_name).mkdir(parents=True, exist_ok=True)
    
    # Get all images
    image_files = sorted(list(input_path.glob('*.png')) + list(input_path.glob('*.jpg')))
    
    if not image_files:
        print(f"No images found in {input_dir}")
        return
    
    print(f"Found {len(image_files)} images")
    print("\nControls:")
    print("  1 - Label as NO_FLOW")
    print("  2 - Label as DRIPPING")
    print("  3 - Label as JETTING")
    print("  4 - Label as CO_FLOW")
    print("  s - Skip image")
    print("  q - Quit")
    print("-" * 50)
    
    labeled_count = 0
    
    for i, img_path in enumerate(image_files):
        # Load and display image
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        
        # Resize for display if too large
        h, w = img.shape[:2]
        if w > 1200:
            scale = 1200 / w
            img = cv2.resize(img, (1200, int(h * scale)))
        
        window_name = f"Label Image ({i+1}/{len(image_files)})"
        cv2.imshow(window_name, img)
        
        print(f"\nImage {i+1}/{len(image_files)}: {img_path.name}")
        
        while True:
            key = cv2.waitKey(0) & 0xFF
            
            if key == ord('1'):
                dest = output_path / 'NO_FLOW' / img_path.name
                shutil.copy(img_path, dest)
                print(f"  -> Labeled as NO_FLOW")
                labeled_count += 1
                break
            elif key == ord('2'):
                dest = output_path / 'DRIPPING' / img_path.name
                shutil.copy(img_path, dest)
                print(f"  -> Labeled as DRIPPING")
                labeled_count += 1
                break
            elif key == ord('3'):
                dest = output_path / 'JETTING' / img_path.name
                shutil.copy(img_path, dest)
                print(f"  -> Labeled as JETTING")
                labeled_count += 1
                break
            elif key == ord('4'):
                dest = output_path / 'CO_FLOW' / img_path.name
                shutil.copy(img_path, dest)
                print(f"  -> Labeled as CO_FLOW")
                labeled_count += 1
                break
            elif key == ord('s'):
                print(f"  -> Skipped")
                break
            elif key == ord('q'):
                cv2.destroyAllWindows()
                print(f"\nLabeled {labeled_count} images")
                return
        
        cv2.destroyAllWindows()
    
    print(f"\nComplete! Labeled {labeled_count} images")
    print(f"Output directory: {output_dir}")
    
    # Show class distribution
    for class_name in ['NO_FLOW', 'DRIPPING', 'JETTING', 'CO_FLOW']:
        count = len(list((output_path / class_name).glob('*')))
        print(f"  {class_name}: {count} images")


def main():
    parser = argparse.ArgumentParser(description='Label images for regime classification')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing images to label')
    parser.add_argument('--output_dir', type=str, default='training_data',
                        help='Output directory for labeled data')
    
    args = parser.parse_args()
    
    label_images(args.input_dir, args.output_dir)


if __name__ == '__main__':
    main()
