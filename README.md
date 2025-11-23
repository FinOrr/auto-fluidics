# Auto-Fluidics

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![GitHub top language](https://img.shields.io/github/languages/top/FinOrr/auto-fluidics.svg)](https://github.com/FinOrr/auto-fluidics)

Python toolkit for automated droplet detection and flow regime classification in microfluidic experiments using computer vision and machine learning.

## Quick Start

```bash
git clone git@github.com:FinOrr/auto-fluidics.git
cd auto-fluidics
pip install -r requirements.txt
streamlit run app.py
```

## Features

- Droplet detection using Hough Circle Transform
- Flow regime classification: DRIPPING, JETTING, CO_FLOW, NO_FLOW
- ML classifier using MobileNetV2
- Real-time metrics: count, size distribution, uniformity, aspect ratio
- Interactive Streamlit dashboard and batch processing
- Processing speed: 10-100ms per image

## Usage

### Dashboard

```bash
streamlit run app.py
```

### Python API

```python
from processing.particle_detector import ParticleImageProcessor
from perception.regime_detector import RegimeDetector

processor = ParticleImageProcessor(um_per_pixel=2.0)
processor.process_image('img/sample-10.png')

detector = RegimeDetector()
regime, confidence, _ = detector.detect_regime(processor.particle_metrics)

print(f"Particles: {processor.particle_metrics['num_particles']}")
print(f"Regime: {regime.name} ({confidence:.0%})")
```

## Key Files

**Main:**
- `app.py` - Streamlit dashboard
- `validate_detection.py` - Batch validation

**ML:**
- `train_regime_classifier.py` - Train ML classifier
- `label_training_data.py` - Label training images
- `check_ml_setup.py` - Verify TensorFlow setup

**Video:**
- `batch_video_extract.py` - Extract frames from videos
- `extract_training_from_video.py` - Extract and label video frames

**Core:**
- `processing/particle_detector.py` - Droplet detection engine
- `processing/channel_detector.py` - Channel ROI extraction
- `perception/regime_detector.py` - Rule-based regime classification
- `perception/ml_regime_classifier.py` - ML-based classification

## How It Works

**Detection:** Hough Circle Transform with adaptive parameters, watermark filtering, and outlier removal.

**Classification:** Rule-based uses particle count, aspect ratio, and size uniformity. ML classifier uses MobileNetV2 fine-tuned on labeled microfluidic images.

## License

MIT License - see [LICENSE](LICENSE)
