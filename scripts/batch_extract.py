#!/usr/bin/env python3
"""
Batch video processing with configuration file.

Uses a simple YAML config file to process multiple videos.

Config file format (videos.yaml):
---
output_dir: training_data
target_size: 224
fps: 2.0
max_frames_per_video: 100
min_quality: 20.0

videos:
  - path: videos/dripping_highflow.mp4
    label: DRIPPING
  - path: videos/dripping_lowflow.mp4
    label: DRIPPING
  - path: videos/jetting_unstable.mp4
    label: JETTING
  - path: videos/noflow_empty.mp4
    label: NO_FLOW

Usage:
    python batch_video_extract.py --config videos.yaml

Author: Auto-Fluidics Project
"""

import argparse
import yaml
from pathlib import Path
from extract_training_from_video import VideoFrameExtractor


def process_config(config_path: str):
    """Process videos from YAML config file."""
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Get settings
    output_dir = config.get('output_dir', 'training_data')
    target_size = config.get('target_size', 224)
    fps = config.get('fps', 2.0)
    max_frames = config.get('max_frames_per_video', None)
    min_quality = config.get('min_quality', 20.0)
    
    videos = config.get('videos', [])
    
    if not videos:
        raise ValueError("No videos found in config file")
    
    print("Configuration:")
    print(f"  Output dir: {output_dir}")
    print(f"  Target size: {target_size}x{target_size}")
    print(f"  FPS: {fps}")
    print(f"  Max frames per video: {max_frames or 'unlimited'}")
    print(f"  Min quality: {min_quality}")
    print(f"  Videos: {len(videos)}")
    print()
    
    # Create extractor
    extractor = VideoFrameExtractor(target_size=target_size, min_quality=min_quality)
    
    # Process each video
    total_extracted = 0
    class_counts = {'NO_FLOW': 0, 'DRIPPING': 0, 'JETTING': 0, 'CO_FLOW': 0}
    
    for i, video_entry in enumerate(videos, 1):
        video_path = video_entry.get('path')
        label = video_entry.get('label')
        
        if not video_path or not label:
            print(f"Skipping invalid entry: {video_entry}")
            continue
        
        if not Path(video_path).exists():
            print(f"Video not found: {video_path}")
            continue
        
        print(f"\n[{i}/{len(videos)}] {'='*60}")
        print(f"Processing: {video_path} -> {label}")
        print('='*60)
        
        try:
            count = extractor.extract_frames(
                video_path, output_dir, label,
                fps=fps, max_frames=max_frames
            )
            total_extracted += count
            class_counts[label] += count
        except Exception as e:
            print(f"Error processing {video_path}: {e}")
            continue
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print('='*60)
    print(f"Total frames extracted: {total_extracted}")
    print(f"Output directory: {output_dir}")
    print("\nClass distribution:")
    for label, count in class_counts.items():
        if count > 0:
            print(f"  {label}: {count} images")
    print('='*60)


def create_example_config(output_path: str):
    """Create an example config file."""
    
    example_config = {
        'output_dir': 'training_data',
        'target_size': 224,
        'fps': 2.0,
        'max_frames_per_video': 100,
        'min_quality': 20.0,
        'videos': [
            {'path': 'videos/dripping_example1.mp4', 'label': 'DRIPPING'},
            {'path': 'videos/dripping_example2.mp4', 'label': 'DRIPPING'},
            {'path': 'videos/jetting_example.mp4', 'label': 'JETTING'},
            {'path': 'videos/noflow_example.mp4', 'label': 'NO_FLOW'},
            {'path': 'videos/coflow_example.mp4', 'label': 'CO_FLOW'},
        ]
    }
    
    with open(output_path, 'w') as f:
        yaml.dump(example_config, f, default_flow_style=False, sort_keys=False)
    
    print(f"Example config created: {output_path}")
    print("Edit this file with your video paths and run:")
    print(f"  python batch_video_extract.py --config {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Batch video processing from config file')
    parser.add_argument('--config', type=str, help='Path to YAML config file')
    parser.add_argument('--create_example', type=str,
                        help='Create example config file at specified path')
    
    args = parser.parse_args()
    
    if args.create_example:
        create_example_config(args.create_example)
    elif args.config:
        process_config(args.config)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
