#!/usr/bin/env python3
"""
Check ML classifier setup and provide guidance.

Usage:
    python check_ml_setup.py
"""

import sys
from pathlib import Path


def check_dependencies():
    """Check if required packages are installed."""
    print("Checking dependencies...")
    
    packages = {
        'tensorflow': 'TensorFlow',
        'cv2': 'OpenCV',
        'numpy': 'NumPy',
        'yaml': 'PyYAML',
    }
    
    missing = []
    
    for module, name in packages.items():
        try:
            __import__(module)
            print(f"  OK:  {name}")
        except ImportError:
            print(f"  ERR: {name} - NOT INSTALLED")
            missing.append(name)
    
    if missing:
        print(f"\nMissing packages: {', '.join(missing)}")
        print("Install with: pip install tensorflow matplotlib pyyaml")
        return False
    
    print("\nOK:  All dependencies installed")
    return True


def check_training_data():
    """Check if training data exists."""
    print("\nChecking training data...")
    
    data_dir = Path('training_data')
    
    if not data_dir.exists():
        print("  ERR: training_data/ directory not found")
        print("\nNext steps:")
        print("  1. Extract from videos:")
        print("     python extract_training_from_video.py --video VIDEO.mp4 --label DRIPPING --roi")
        print("  2. Or use interactive labeling:")
        print("     python label_training_data.py --input_dir img --output_dir training_data")
        return False
    
    classes = ['NO_FLOW', 'DRIPPING', 'JETTING', 'CO_FLOW']
    total_images = 0
    class_counts = {}
    
    for class_name in classes:
        class_dir = data_dir / class_name
        if class_dir.exists():
            count = len(list(class_dir.glob('*.png'))) + len(list(class_dir.glob('*.jpg')))
            class_counts[class_name] = count
            total_images += count
            
            if count > 0:
                status = "OK: " if count >= 50 else "⚠"
                print(f"  {status} {class_name}: {count} images" + 
                      (f" (recommend 50+)" if count < 50 else ""))
            else:
                print(f"  ERR: {class_name}: no images")
        else:
            print(f"  ERR: {class_name}: directory not found")
            class_counts[class_name] = 0
    
    if total_images == 0:
        print("\nERR: No training data found")
        return False
    elif total_images < 150:
        print(f"\n⚠ Only {total_images} images total (recommend 150+ for good results)")
        return True
    else:
        print(f"\nOK:  Found {total_images} training images")
        return True


def check_model():
    """Check if trained model exists."""
    print("\nChecking trained model...")
    
    model_path = Path('models/regime_classifier.weights.h5')
    
    if not model_path.exists():
        print("  ERR: Trained model not found")
        print("\nNext step:")
        print("  python train_regime_classifier.py --data_dir training_data --epochs 20")
        return False
    else:
        size_mb = model_path.stat().st_size / (1024 * 1024)
        print(f"  OK:  Model found ({size_mb:.1f} MB)")
        return True


def check_videos():
    """Check if videos directory exists."""
    print("\nChecking for videos...")
    
    videos_dir = Path('videos')
    
    if not videos_dir.exists():
        print("  ℹ No videos/ directory")
        print("  (Optional: Create videos/ and place your experiment videos there)")
        return None
    
    video_files = (
        list(videos_dir.glob('*.mp4')) +
        list(videos_dir.glob('*.avi')) +
        list(videos_dir.glob('*.mov'))
    )
    
    if not video_files:
        print("  ℹ videos/ directory is empty")
        return None
    
    print(f"  OK:  Found {len(video_files)} video(s)")
    for video in video_files[:5]:  # Show first 5
        print(f"    - {video.name}")
    if len(video_files) > 5:
        print(f"    ... and {len(video_files) - 5} more")
    
    return True


def provide_guidance(has_deps, has_data, has_model):
    """Provide next steps based on current state."""
    print("\n" + "="*60)
    print("NEXT STEPS")
    print("="*60)
    
    if not has_deps:
        print("\n1. Install dependencies:")
        print("   pip install tensorflow matplotlib pyyaml")
        return
    
    if not has_data:
        print("\n1. Generate training data:")
        print("   Option A - From videos (recommended):")
        print("     python extract_training_from_video.py --video VIDEO.mp4 --label DRIPPING --roi")
        print("   Option B - Interactive labeling:")
        print("     python label_training_data.py --input_dir img")
        return
    
    if not has_model:
        print("\n1. Train the model:")
        print("   python train_regime_classifier.py --data_dir training_data --epochs 20")
        print("\n2. Then run the app:")
        print("   streamlit run app.py")
        return
    
    print("\nOK:  Setup complete!")
    print("\nRun the app:")
    print("  streamlit run app.py")
    print("\nIn the app:")
    print("  1. Sidebar → 'Flow Classification'")
    print("  2. Enable regime classification")
    print("  3. Select 'ML (CNN)' classifier")


def main():
    print("="*60)
    print("ML Classifier Setup Check")
    print("="*60)
    
    has_deps = check_dependencies()
    has_data = check_training_data()
    has_model = check_model()
    check_videos()
    
    provide_guidance(has_deps, has_data, has_model)
    
    print("\n" + "="*60)
    print("For detailed guides, see:")
    print("  - ML_QUICKSTART.md")
    print("  - VIDEO_EXTRACTION_GUIDE.md")
    print("  - ML_CLASSIFIER_README.md")
    print("="*60)


if __name__ == '__main__':
    main()
