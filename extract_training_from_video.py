#!/usr/bin/env python3
"""
Extract training images from videos for regime classification.

Extracts frames from videos and automatically organizes them into training data structure.
Handles resolution normalization and quality filtering.

Usage:
    python extract_training_from_video.py \\
        --video videos/dripping.mp4 \\
        --label DRIPPING \\
        --output training_data \\
        --fps 2 \\
        --target_size 224

This will extract 2 frames per second from the video and save them to:
    training_data/DRIPPING/dripping_frame_0001.png
    training_data/DRIPPING/dripping_frame_0002.png
    ...

Author: Auto-Fluidics Project
"""

import argparse
import cv2
from pathlib import Path
import numpy as np
from typing import Optional, Tuple


class VideoFrameExtractor:
    """Extract and preprocess frames from videos for ML training."""
    
    def __init__(self, target_size: int = 224, min_quality: float = 20.0):
        """
        Initialize extractor.
        
        Args:
            target_size: Target size for extracted frames (e.g., 224 for 224x224)
            min_quality: Minimum blurriness threshold (higher = sharper required)
        """
        self.target_size = target_size
        self.min_quality = min_quality
    
    def calculate_blurriness(self, image: np.ndarray) -> float:
        """
        Calculate image blurriness using Laplacian variance.
        Higher values = sharper image.
        
        Args:
            image: Input image (BGR)
            
        Returns:
            Blurriness score
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = laplacian.var()
        return variance
    
    def preprocess_frame(self, frame: np.ndarray, roi: Optional[Tuple] = None) -> np.ndarray:
        """
        Preprocess frame for training.
        
        Args:
            frame: Input frame (BGR)
            roi: Optional (x, y, w, h) region of interest to crop
            
        Returns:
            Preprocessed frame
        """
        # Crop to ROI if provided
        if roi is not None:
            x, y, w, h = roi
            frame = frame[y:y+h, x:x+w]
        
        # Resize maintaining aspect ratio, then center crop to square
        h, w = frame.shape[:2]
        
        # Resize so smaller dimension matches target_size
        if h < w:
            new_h = self.target_size
            new_w = int(w * (self.target_size / h))
        else:
            new_w = self.target_size
            new_h = int(h * (self.target_size / w))
        
        frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Center crop to square
        h, w = frame.shape[:2]
        start_y = (h - self.target_size) // 2
        start_x = (w - self.target_size) // 2
        frame = frame[start_y:start_y+self.target_size, start_x:start_x+self.target_size]
        
        return frame
    
    def extract_frames(self, video_path: str, output_dir: str, label: str,
                       fps: Optional[float] = None, max_frames: Optional[int] = None,
                       roi: Optional[Tuple] = None, skip_blurry: bool = True,
                       show_preview: bool = False) -> int:
        """
        Extract frames from video.
        
        Args:
            video_path: Path to input video
            output_dir: Output directory for frames
            label: Class label (NO_FLOW, DRIPPING, JETTING, CO_FLOW)
            fps: Extract at this FPS (None = extract all frames)
            max_frames: Maximum frames to extract (None = no limit)
            roi: Optional (x, y, w, h) region to crop
            skip_blurry: Skip frames below quality threshold
            show_preview: Show preview window during extraction
            
        Returns:
            Number of frames extracted
        """
        video_path = Path(video_path)
        output_path = Path(output_dir) / label
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / video_fps if video_fps > 0 else 0
        
        print(f"Video: {video_path.name}")
        print(f"  FPS: {video_fps:.2f}")
        print(f"  Total frames: {total_frames}")
        print(f"  Duration: {duration:.1f}s")
        
        # Calculate frame skip
        if fps is not None:
            frame_skip = int(video_fps / fps)
            expected_frames = int(duration * fps)
        else:
            frame_skip = 1
            expected_frames = total_frames
        
        if max_frames:
            expected_frames = min(expected_frames, max_frames)
        
        print(f"  Extracting ~{expected_frames} frames (1 every {frame_skip} frames)")
        
        # Extract frames
        frame_count = 0
        saved_count = 0
        skipped_blur = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Check if we should process this frame
            if frame_count % frame_skip != 0:
                frame_count += 1
                continue
            
            # Check max frames limit
            if max_frames and saved_count >= max_frames:
                break
            
            # Check quality
            if skip_blurry:
                blur_score = self.calculate_blurriness(frame)
                if blur_score < self.min_quality:
                    skipped_blur += 1
                    frame_count += 1
                    continue
            
            # Preprocess frame
            processed = self.preprocess_frame(frame, roi)
            
            # Save frame
            base_name = video_path.stem
            frame_name = f"{base_name}_frame_{saved_count:04d}.png"
            output_file = output_path / frame_name
            cv2.imwrite(str(output_file), processed)
            
            saved_count += 1
            
            # Show preview
            if show_preview:
                cv2.imshow(f"Extracting: {label}", processed)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            frame_count += 1
        
        cap.release()
        if show_preview:
            cv2.destroyAllWindows()
        
        print(f"  Extracted: {saved_count} frames")
        if skip_blurry:
            print(f"  Skipped (blurry): {skipped_blur} frames")
        
        return saved_count
    
    def extract_with_roi_selection(self, video_path: str, output_dir: str, label: str,
                                   fps: Optional[float] = 2, max_frames: Optional[int] = None) -> int:
        """
        Extract frames with interactive ROI selection.
        
        Shows first frame and lets you select channel region.
        
        Args:
            video_path: Path to input video
            output_dir: Output directory for frames
            label: Class label
            fps: Extract at this FPS
            max_frames: Maximum frames to extract
            
        Returns:
            Number of frames extracted
        """
        # Open video and get first frame
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            raise ValueError("Could not read first frame")
        
        # Let user select ROI
        print("\n=== ROI Selection ===")
        print("Draw a box around the channel region, then press ENTER")
        print("Press 'c' to cancel and use full frame")
        
        roi = cv2.selectROI("Select Channel Region (ENTER to confirm, c to cancel)", 
                           frame, showCrosshair=True, fromCenter=False)
        cv2.destroyAllWindows()
        
        # Check if ROI was selected
        if roi == (0, 0, 0, 0) or roi[2] == 0 or roi[3] == 0:
            print("No ROI selected - using full frame")
            roi = None
        else:
            print(f"ROI selected: x={roi[0]}, y={roi[1]}, w={roi[2]}, h={roi[3]}")
        
        # Extract frames with ROI
        return self.extract_frames(video_path, output_dir, label, 
                                   fps=fps, max_frames=max_frames, roi=roi)


def batch_extract(video_config: dict, output_dir: str, target_size: int = 224,
                 fps: float = 2.0, max_frames_per_video: Optional[int] = None):
    """
    Extract from multiple videos in batch.
    
    Args:
        video_config: Dict mapping video paths to labels
            e.g., {'videos/dripping1.mp4': 'DRIPPING', 'videos/jetting.mp4': 'JETTING'}
        output_dir: Output directory
        target_size: Target image size
        fps: Extraction FPS
        max_frames_per_video: Max frames per video
    """
    extractor = VideoFrameExtractor(target_size=target_size)
    
    total_extracted = 0
    
    for video_path, label in video_config.items():
        print(f"\n{'='*60}")
        print(f"Processing: {video_path} -> {label}")
        print('='*60)
        
        count = extractor.extract_frames(
            video_path, output_dir, label,
            fps=fps, max_frames=max_frames_per_video
        )
        total_extracted += count
    
    print(f"\n{'='*60}")
    print(f"Total frames extracted: {total_extracted}")
    print(f"Output directory: {output_dir}")
    print('='*60)
    
    # Show class distribution
    output_path = Path(output_dir)
    for class_name in ['NO_FLOW', 'DRIPPING', 'JETTING']:
        class_dir = output_path / class_name
        if class_dir.exists():
            count = len(list(class_dir.glob('*.png')))
            print(f"  {class_name}: {count} images")


def main():
    parser = argparse.ArgumentParser(
        description='Extract training images from videos',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract from single video with ROI selection
  python extract_training_from_video.py --video dripping.mp4 --label DRIPPING --roi

  # Extract at 1 FPS, max 100 frames
  python extract_training_from_video.py --video jetting.mp4 --label JETTING --fps 1 --max_frames 100

  # Batch process multiple videos
  python extract_training_from_video.py --batch \\
      --config "dripping1.mp4:DRIPPING,jetting.mp4:JETTING,noflow.mp4:NO_FLOW"
        """
    )
    
    parser.add_argument('--video', type=str, help='Path to input video')
    parser.add_argument('--label', type=str, choices=['NO_FLOW', 'DRIPPING', 'JETTING', 'CO_FLOW'],
                        help='Class label for extracted frames')
    parser.add_argument('--output', type=str, default='training_data',
                        help='Output directory (default: training_data)')
    parser.add_argument('--fps', type=float, default=2.0,
                        help='Extract frames at this FPS (default: 2.0)')
    parser.add_argument('--max_frames', type=int, default=None,
                        help='Maximum frames to extract per video')
    parser.add_argument('--target_size', type=int, default=224,
                        help='Target size for extracted frames (default: 224)')
    parser.add_argument('--min_quality', type=float, default=20.0,
                        help='Minimum sharpness threshold (default: 20.0)')
    parser.add_argument('--roi', action='store_true',
                        help='Enable interactive ROI selection')
    parser.add_argument('--preview', action='store_true',
                        help='Show preview during extraction')
    parser.add_argument('--batch', action='store_true',
                        help='Batch mode - process multiple videos')
    parser.add_argument('--config', type=str,
                        help='Batch config: "video1.mp4:LABEL1,video2.mp4:LABEL2,..."')
    
    args = parser.parse_args()
    
    extractor = VideoFrameExtractor(
        target_size=args.target_size,
        min_quality=args.min_quality
    )
    
    if args.batch:
        # Batch mode
        if not args.config:
            raise ValueError("--config required for batch mode")
        
        # Parse config
        video_config = {}
        for pair in args.config.split(','):
            video, label = pair.strip().split(':')
            video_config[video] = label.strip()
        
        batch_extract(
            video_config, args.output,
            target_size=args.target_size,
            fps=args.fps,
            max_frames_per_video=args.max_frames
        )
    else:
        # Single video mode
        if not args.video or not args.label:
            raise ValueError("--video and --label required for single mode")
        
        if args.roi:
            # Interactive ROI selection
            count = extractor.extract_with_roi_selection(
                args.video, args.output, args.label,
                fps=args.fps, max_frames=args.max_frames
            )
        else:
            # Extract without ROI
            count = extractor.extract_frames(
                args.video, args.output, args.label,
                fps=args.fps, max_frames=args.max_frames,
                show_preview=args.preview
            )
        
        print(f"\nExtracted {count} frames to {args.output}/{args.label}/")


if __name__ == '__main__':
    main()
