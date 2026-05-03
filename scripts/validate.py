#!/usr/bin/env python3
"""
Validation Script for Particle Detection and Regime Classification

Tests the perception and processing modules on sample images from img/ folder.
Outputs metrics and annotated images to validate detection accuracy.
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bacha_vision.detection.droplet_detector import ParticleImageProcessor
from bacha_vision.classification.regime_detector import RegimeDetector, regime_to_string

def validate_all_images(img_dir='img', output_dir='validation_output'):
    """
    Run validation on all images in img_dir

    Args:
        img_dir: Directory containing sample images
        output_dir: Directory to save annotated outputs
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Initialize processor and detector
    processor = ParticleImageProcessor()
    regime_detector = RegimeDetector()

    # Get all PNG images
    img_paths = sorted(Path(img_dir).glob('*.png'))

    print("=" * 80)
    print("PARTICLE DETECTION VALIDATION")
    print("=" * 80)
    print(f"Found {len(img_paths)} images to process\n")

    results = []

    for img_path in img_paths:
        print(f"\n{'─' * 80}")
        print(f"Processing: {img_path.name}")
        print('─' * 80)

        # Process image
        processor.process_image(str(img_path))

        # Get metrics
        metrics = processor.particle_metrics

        # Detect regime
        regime, confidence, indicators = regime_detector.detect_regime(metrics)

        # Print results
        print(f"\n📊 Detection Results:")
        print(f"  Particles detected:    {metrics['num_particles']}")

        # Handle None values for when no particles detected
        mean_size = metrics['mean_particle_size']
        median_size = metrics['median_particle_size']
        std_size = metrics['std_dev_particle_size']
        aspect = metrics['mean_aspect_ratio']
        cv = metrics['coefficient_of_variation']

        print(f"  Mean size:             {mean_size:.1f} pixels" if mean_size and not np.isnan(mean_size) else "  Mean size:             N/A")
        print(f"  Median size:           {median_size:.1f} pixels" if median_size else "  Median size:           N/A")
        print(f"  Std dev:               {std_size:.1f} pixels" if std_size and not np.isnan(std_size) else "  Std dev:               N/A")
        print(f"  Mean aspect ratio:     {aspect:.2f}" if aspect and not np.isnan(aspect) else "  Mean aspect ratio:     N/A")
        print(f"  Coefficient of Var:    {cv:.3f}" if cv and not np.isnan(cv) else "  Coefficient of Var:    N/A")

        print(f"\n🔬 Regime Classification:")
        print(f"  Regime:                {regime_to_string(regime)}")
        print(f"  Confidence:            {confidence:.2f}")
        print(f"  Aspect ratio thresh:   {indicators['aspect_ratio_threshold']:.2f}")
        print(f"  Size CV thresh:        {indicators['size_cv_threshold']:.2f}")

        # Save annotated image
        annotated = processor.image.get('processed')
        if annotated is not None:
            # Add regime text overlay
            annotated_copy = annotated.copy()
            regime_text = f"{regime_to_string(regime)} ({confidence:.0%})"

            # Add text background for readability
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.0
            thickness = 2

            # Get text size
            (text_w, text_h), baseline = cv2.getTextSize(regime_text, font, font_scale, thickness)

            # Draw background rectangle
            cv2.rectangle(annotated_copy, (10, 10), (20 + text_w, 30 + text_h), (0, 0, 0), -1)

            # Draw text
            color_map = {
                'NO FLOW': (0, 0, 255),      # Red
                'DRIPPING': (0, 255, 0),     # Green
                'JETTING': (0, 165, 255),    # Orange
                'UNKNOWN': (255, 255, 255)   # White
            }
            color = color_map.get(regime_to_string(regime), (255, 255, 255))
            cv2.putText(annotated_copy, regime_text, (15, 25 + text_h),
                       font, font_scale, color, thickness)

            # Save
            output_path = os.path.join(output_dir, f"annotated_{img_path.name}")
            cv2.imwrite(output_path, annotated_copy)
            print(f"\n💾 Saved: {output_path}")

        # Store results (use 0 for None/nan values)
        results.append({
            'filename': img_path.name,
            'num_particles': metrics['num_particles'] or 0,
            'mean_size': mean_size if mean_size and not np.isnan(mean_size) else 0,
            'median_size': median_size if median_size else 0,
            'std_dev': std_size if std_size and not np.isnan(std_size) else 0,
            'mean_aspect_ratio': aspect if aspect and not np.isnan(aspect) else 1.0,
            'cv': cv if cv and not np.isnan(cv) else 0,
            'regime': regime_to_string(regime),
            'confidence': confidence
        })

    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

    regime_counts = {}
    for r in results:
        regime_counts[r['regime']] = regime_counts.get(r['regime'], 0) + 1

    print(f"\nRegime Distribution:")
    for regime, count in sorted(regime_counts.items()):
        print(f"  {regime:15s}: {count:2d} images ({count/len(results)*100:.1f}%)")

    print(f"\nParticle Count Statistics:")
    counts = [r['num_particles'] for r in results]
    print(f"  Mean:   {np.mean(counts):.1f}")
    print(f"  Median: {np.median(counts):.1f}")
    print(f"  Min:    {np.min(counts)}")
    print(f"  Max:    {np.max(counts)}")

    print(f"\nSize Statistics (pixels):")
    sizes = [r['mean_size'] for r in results if r['num_particles'] > 0]
    if sizes:
        print(f"  Mean:   {np.mean(sizes):.1f}")
        print(f"  Median: {np.median(sizes):.1f}")
        print(f"  Std:    {np.std(sizes):.1f}")

    print(f"\nAspect Ratio Statistics:")
    aspect_ratios = [r['mean_aspect_ratio'] for r in results if r['num_particles'] > 0]
    if aspect_ratios:
        print(f"  Mean:   {np.mean(aspect_ratios):.2f}")
        print(f"  Median: {np.median(aspect_ratios):.2f}")
        print(f"  Max:    {np.max(aspect_ratios):.2f}")

    print(f"\n✅ Validation complete! Annotated images saved to: {output_dir}/")
    print("=" * 80)

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Validate particle detection on sample images')
    parser.add_argument('--img-dir', default='img', help='Directory containing sample images')
    parser.add_argument('--output-dir', default='validation_output', help='Output directory for annotated images')

    args = parser.parse_args()

    results = validate_all_images(args.img_dir, args.output_dir)
