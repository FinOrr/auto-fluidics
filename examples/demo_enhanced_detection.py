#!/usr/bin/env python3
"""
Demo script showing enhanced detection capabilities including:
- Aspect ratio detection
- Regime classification
- Encapsulation analysis
- Temporal tracking

Usage:
    python demo_enhanced_detection.py /path/to/image.png
    python demo_enhanced_detection.py --stream 192.168.1.34
"""

import sys
import os
import cv2

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from processing.particle_detector import ParticleImageProcessor
from perception.regime_detector import RegimeDetector, regime_to_string, is_safe_regime


def demo_single_image(image_path, um_per_pixel=None):
    """Demonstrate enhanced detection on a single image."""

    print("=" * 60)
    print("Enhanced Particle Detection Demo")
    print("=" * 60)

    # Initialize processor with calibration
    processor = ParticleImageProcessor(um_per_pixel=um_per_pixel)

    # Enable encapsulation detection
    try:
        processor.enable_encapsulation_detection(True)
        print("OK:  Encapsulation detection enabled")
    except ImportError:
        print("⚠ Encapsulation detection not available")

    # Initialize regime detector
    regime_detector = RegimeDetector()

    # Process image
    print(f"\nProcessing: {image_path}")
    processor.process_image(image_path)

    # Display enhanced metrics
    print("\n" + "=" * 60)
    print("ENHANCED METRICS")
    print("=" * 60)

    pm = processor.particle_metrics

    print(f"\nBasic Detection:")
    print(f"  Particles detected: {pm['num_particles']}")
    print(f"  Mean size: {pm['mean_particle_size']:.2f} px")
    if um_per_pixel:
        print(f"  Mean size: {pm['mean_particle_size'] * um_per_pixel:.2f} µm")

    print(f"\nUniformity Metrics:")
    print(f"  Std deviation: {pm['std_dev_particle_size']:.2f} px")
    print(f"  CV (coefficient of variation): {pm['coefficient_of_variation']:.3f}")
    if pm['coefficient_of_variation'] < 0.05:
        print(f"  → Excellent uniformity (CV < 5%)")
    elif pm['coefficient_of_variation'] < 0.10:
        print(f"  → Good uniformity (CV < 10%)")
    else:
        print(f"  → High variation (CV > 10%)")

    print(f"\nShape Analysis:")
    print(f"  Mean aspect ratio: {pm['mean_aspect_ratio']:.2f}")
    if pm['mean_aspect_ratio'] > 2.5:
        print(f"  → WARNING: Elongated shapes detected (possible jetting)")
    elif pm['mean_aspect_ratio'] > 1.5:
        print(f"  → Moderately elongated")
    else:
        print(f"  → Circular (aspect ratio ~ 1.0)")

    # Regime detection
    regime, confidence, indicators = regime_detector.detect_regime(pm)

    print(f"\nRegime Classification:")
    print(f"  Regime: {regime_to_string(regime)}")
    print(f"  Confidence: {confidence:.2f}")
    print(f"  Safe for control: {'OK:  YES' if is_safe_regime(regime) else 'ERR: NO'}")

    # Encapsulation metrics (if enabled)
    if pm['encapsulation_rate'] is not None:
        print(f"\nEncapsulation Analysis:")
        print(f"  Droplets detected: {pm['num_droplets']}")
        print(f"  Encapsulation rate: {pm['encapsulation_rate']*100:.1f}%")
        print(f"  Particles per droplet: {pm['particles_per_droplet']:.2f}")
        print(f"  Poisson λ: {pm['poisson_lambda']:.2f}")

        if pm['encapsulation_rate'] > 0.9:
            print(f"  → Excellent encapsulation (>90%)")
        elif pm['encapsulation_rate'] > 0.7:
            print(f"  → Good encapsulation (>70%)")
        else:
            print(f"  → Low encapsulation (<70%)")

    print("\n" + "=" * 60)

    # Show result
    if processor.image['processed'] is not None:
        cv2.imshow('Enhanced Detection Result', processor.image['processed'])
        print("\nPress any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def demo_stream(stream_ip, um_per_pixel=None):
    """Demonstrate real-time regime detection on video stream."""

    print("=" * 60)
    print("Real-Time Regime Detection Demo")
    print("=" * 60)
    print(f"Stream: {stream_ip}")
    print("Press ESC to exit")
    print("=" * 60)

    # Initialize
    processor = ParticleImageProcessor(um_per_pixel=um_per_pixel)
    regime_detector = RegimeDetector()

    # Try to enable encapsulation
    try:
        processor.enable_encapsulation_detection(True)
    except ImportError:
        pass

    frame_count = 0

    while cv2.waitKey(1) != 27:  # ESC to exit
        # Process frame
        processor.process_stream(stream_ip)
        frame_count += 1

        # Detect regime
        regime, confidence, indicators = regime_detector.detect_regime(
            processor.particle_metrics
        )

        # Display info every 10 frames
        if frame_count % 10 == 0:
            pm = processor.particle_metrics
            print(f"\nFrame {frame_count}:")
            print(f"  Particles: {pm['num_particles']}")
            print(f"  Mean size: {pm['mean_particle_size']:.1f} px")
            print(f"  CV: {pm['coefficient_of_variation']:.3f}")
            print(f"  Aspect ratio: {pm['mean_aspect_ratio']:.2f}")
            print(f"  Regime: {regime_to_string(regime)} ({confidence:.2f})")

            if not is_safe_regime(regime):
                print(f"  ⚠ WARNING: Unsafe regime!")

            # Regime stability
            stability = regime_detector.get_regime_stability(window_size=10)
            print(f"  Stability: {stability*100:.0f}%")

        # Display
        if processor.image['processed'] is not None:
            # Add regime indicator on image
            img = processor.image['processed'].copy()
            regime_text = f"{regime_to_string(regime)} ({confidence:.2f})"
            color = (0, 255, 0) if is_safe_regime(regime) else (0, 0, 255)

            cv2.putText(img, regime_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            cv2.imshow('Real-Time Detection', img)

    cv2.destroyAllWindows()
    print(f"\nProcessed {frame_count} frames")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Demo enhanced particle detection")
    parser.add_argument('--stream', type=str, help='Stream IP address for real-time demo')
    parser.add_argument('--image', type=str, help='Path to image file for single-image demo')
    parser.add_argument('--calibration', type=float, help='Calibration (µm/pixel)')
    parser.add_argument('input', nargs='?', help='Image path or stream IP (auto-detected)')

    args = parser.parse_args()

    # Auto-detect input type
    if args.input:
        if os.path.isfile(args.input):
            args.image = args.input
        else:
            args.stream = args.input

    if args.image:
        demo_single_image(args.image, um_per_pixel=args.calibration)
    elif args.stream:
        demo_stream(args.stream, um_per_pixel=args.calibration)
    else:
        parser.print_help()
        print("\nExamples:")
        print("  python demo_enhanced_detection.py /path/to/image.png")
        print("  python demo_enhanced_detection.py --stream 192.168.1.34")
        print("  python demo_enhanced_detection.py /path/to/image.png --calibration 2.0")
