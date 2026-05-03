#!/usr/bin/env python3
"""
Auto-Fluidics: Interactive Dashboard

Streamlit app for real-time droplet detection and analysis.
"""

import streamlit as st
import cv2
import numpy as np
from pathlib import Path
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bacha_vision.detection.droplet_detector import ParticleImageProcessor
from bacha_vision.classification.regime_detector import RegimeDetector, regime_to_string, FlowRegime

# Try to import ML classifier
try:
    from bacha_vision.classification.ml_classifier import MLRegimeClassifier
    ML_CLASSIFIER_AVAILABLE = True
except ImportError:
    ML_CLASSIFIER_AVAILABLE = False
    print("ML classifier not available. Install TensorFlow: pip install tensorflow")

# Page config
st.set_page_config(
    page_title="Auto-Fluidics Dashboard",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.metric-card {
    background-color: #f0f2f6;
    padding: 15px;
    border-radius: 10px;
    margin: 10px 0;
}
.section-header {
    font-size: 18px;
    font-weight: 600;
    margin-top: 20px;
    margin-bottom: 10px;
    color: #1f77b4;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'processor' not in st.session_state:
    st.session_state.processor = ParticleImageProcessor()
if 'regime_detector' not in st.session_state:
    st.session_state.regime_detector = RegimeDetector()
if 'ml_classifier' not in st.session_state:
    if ML_CLASSIFIER_AVAILABLE:
        # Try to load trained model
        model_path = Path('models/regime_classifier.weights.h5')
        if model_path.exists():
            st.session_state.ml_classifier = MLRegimeClassifier(str(model_path))
        else:
            st.session_state.ml_classifier = None
    else:
        st.session_state.ml_classifier = None
if 'last_processed_image' not in st.session_state:
    st.session_state.last_processed_image = None
if 'enable_regime_classification' not in st.session_state:
    st.session_state.enable_regime_classification = False
if 'use_ml_classifier' not in st.session_state:
    st.session_state.use_ml_classifier = True
if 'detection_config' not in st.session_state:
    st.session_state.detection_config = {
        'dp': 1,
        'minDist': 40,
        'param1': 70,
        'param2': 50,
        'minRadius': 10,
        'maxRadius': 500,
        'use_custom': False
    }
if 'last_config' not in st.session_state:
    st.session_state.last_config = None


def draw_clean_annotations(image, centres, radii):
    """Draw only circles on image, no text overlay"""
    annotated = image.copy()

    GREEN = (0, 255, 0)

    for centre, radius in zip(centres, radii):
        cv2.circle(annotated, centre, radius, GREEN, 2)
        # Draw center point
        cv2.circle(annotated, centre, 3, GREEN, -1)

    return annotated


def process_and_display(image_path, processor, regime_detector, show_regime=False, detection_config=None, ml_classifier=None, use_ml=True):
    """Process image and display results"""

    # Process image with custom config if provided
    if detection_config and detection_config.get('use_custom'):
        processor.process_image(str(image_path), hough_params={
            'dp': detection_config['dp'],
            'minDist': detection_config['minDist'],
            'param1': detection_config['param1'],
            'param2': detection_config['param2'],
            'minRadius': detection_config['minRadius'],
            'maxRadius': detection_config['maxRadius']
        })
    else:
        # Reset to defaults and use auto-detection
        processor.process_image(str(image_path), hough_params={
            'dp': 1,
            'minDist': 40,
            'param1': None,  # Let auto-detection determine
            'param2': None,  # Let auto-detection determine
            'minRadius': 10,
            'maxRadius': 500
        })
    
    metrics = processor.particle_metrics

    # Detect regime
    regime, confidence, _ = None, 0, {}
    regime_probs = None
    
    if show_regime:
        if use_ml and ml_classifier and ml_classifier.trained:
            # Use ML classifier on the original image
            img = cv2.imread(str(image_path))
            regime, confidence, regime_probs = ml_classifier.predict(img)
        else:
            # Fall back to rule-based
            regime, confidence, _ = regime_detector.detect_regime(metrics)

    # Get clean annotated image (just circles)
    annotated = None
    if processor.image.get('sample') is not None:
        centres = metrics['particle_centres']
        radii = metrics['particle_radii']
        annotated = draw_clean_annotations(processor.image['sample'], centres, radii)

    # Layout with centered content: 1/8 spacing, 6/8 content, 1/8 spacing
    spacer_left, content_col, spacer_right = st.columns([1, 6, 1])

    with content_col:
        # Image display
        st.markdown("### Detection Results")
        if annotated is not None:
            # Convert BGR to RGB
            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            st.image(annotated_rgb, use_container_width=True, caption=f"Detected {metrics['num_particles']} particles")

        st.markdown("---")

        # Summary metrics in grid format
        st.markdown("### Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Particles", metrics['num_particles'],
                     help="Number of droplets detected")
        
        with col2:
            mean_size = metrics.get('mean_particle_size')
            if mean_size and not np.isnan(mean_size):
                unit = "µm" if processor.calibration.get('um_per_pixel') else "px"
                st.metric("Mean Size", f"{mean_size:.1f} {unit}",
                         help="Average particle radius")
            else:
                st.metric("Mean Size", "N/A")
        
        with col3:
            cv = metrics.get('coefficient_of_variation')
            if cv and not np.isnan(cv):
                st.metric("CV", f"{cv:.3f}", help="Coefficient of variation")
            else:
                st.metric("CV", "N/A")
        
        with col4:
            aspect = metrics.get('mean_aspect_ratio')
            if aspect and not np.isnan(aspect):
                st.metric("Aspect Ratio", f"{aspect:.2f}", help="Mean aspect ratio")
            else:
                st.metric("Aspect Ratio", "N/A")

        # Quality indicators
        if metrics['num_particles'] > 0:
            st.markdown("")  # spacing
            
            # Show regime first if enabled (most important)
            if show_regime and regime:
                col_reg = st.columns(1)[0]
                with col_reg:
                    st.markdown("**Flow Regime**")
                    regime_str = regime_to_string(regime)
                    
                    if regime == FlowRegime.DRIPPING:
                        st.success(f"{regime_str} (Confidence: {confidence:.0%})")
                    elif regime == FlowRegime.JETTING:
                        st.error(f"{regime_str} (Confidence: {confidence:.0%})")
                    elif regime == FlowRegime.NO_FLOW:
                        st.warning(f"{regime_str} (Confidence: {confidence:.0%})")
                    elif regime == FlowRegime.CO_FLOW:
                        st.info(f"{regime_str} (Confidence: {confidence:.0%})")
                    else:
                        st.info(f"{regime_str} (Confidence: {confidence:.0%})")
                
                st.markdown("")  # spacing
            
            col_a, col_b = st.columns(2)
            
            with col_a:
                st.markdown("**Uniformity**")
                cv = metrics.get('coefficient_of_variation')
                if cv and not np.isnan(cv):
                    if cv < 0.1:
                        st.success("Excellent uniformity")
                    elif cv < 0.2:
                        st.info("Good uniformity")
                    elif cv < 0.3:
                        st.warning("Fair uniformity")
                    else:
                        st.error("Poor uniformity")
                else:
                    st.caption("N/A")
            
            with col_b:
                st.markdown("**Shape**")
                aspect = metrics.get('mean_aspect_ratio')
                if aspect and not np.isnan(aspect):
                    if aspect < 1.5:
                        st.success("Circular")
                    elif aspect < 2.5:
                        st.info("Slightly elongated")
                    else:
                        st.error("Elongated")
                else:
                    st.caption("N/A")

    with spacer_right:
        pass  # Just spacing

    # Detailed metrics in expander (centered with same layout)
    spacer_left2, detail_col, spacer_right2 = st.columns([1, 6, 1])
    
    with detail_col:
        with st.expander("Detailed Metrics", expanded=False):
            # Size Distribution Metrics
            if metrics['num_particles'] > 1:
                st.markdown("**Size Distribution**")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    median = metrics.get('median_particle_size')
                    if median and not np.isnan(median):
                        unit = "µm" if processor.calibration.get('um_per_pixel') else "px"
                        st.metric("Median", f"{median:.1f} {unit}")
                    else:
                        st.metric("Median", "N/A")
                
                with col2:
                    std = metrics.get('std_dev_particle_size')
                    if std and not np.isnan(std):
                        unit = "µm" if processor.calibration.get('um_per_pixel') else "px"
                        st.metric("Std Dev", f"{std:.1f} {unit}")
                    else:
                        st.metric("Std Dev", "N/A")
                
                with col3:
                    if metrics.get('particle_radii'):
                        min_size = min(metrics['particle_radii'])
                        unit = "µm" if processor.calibration.get('um_per_pixel') else "px"
                        st.metric("Min Size", f"{min_size:.1f} {unit}")
                    else:
                        st.metric("Min Size", "N/A")
                
                with col4:
                    if metrics.get('particle_radii'):
                        max_size = max(metrics['particle_radii'])
                        unit = "µm" if processor.calibration.get('um_per_pixel') else "px"
                        st.metric("Max Size", f"{max_size:.1f} {unit}")
                    else:
                        st.metric("Max Size", "N/A")
                
                st.markdown("")  # spacing

            # Spatial & Coverage Metrics
            st.markdown("**Spatial Analysis**")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                density = metrics.get('particle_density')
                if density and not np.isnan(density):
                    st.metric("Density", f"{density:.6f}", help="Particles per unit area")
                else:
                    st.metric("Density", "N/A")
            
            with col2:
                coverage = metrics.get('particle_coverage')
                if coverage and not np.isnan(coverage):
                    st.metric("Coverage", f"{coverage*100:.2f}%", help="Area covered by particles")
                else:
                    st.metric("Coverage", "N/A")
            
            with col3:
                # Empty column for grid alignment
                st.metric("", "")
            
            with col4:
                # Empty column for grid alignment
                st.metric("", "")
            
            st.markdown("")  # spacing

            # Image Quality Metrics
            st.markdown("**Image Quality**")
            img_meta = processor.image_meta
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                blur = img_meta.get('blurriness', 0)
                if blur is not None and not np.isnan(blur):
                    st.metric("Blurriness", f"{blur:.2f}", help="Lower is better")
                else:
                    st.metric("Blurriness", "N/A", help="Lower is better")
            
            with col2:
                illum = img_meta.get('illumination_uniformity', 0)
                if illum is not None and not np.isnan(illum):
                    st.metric("Illumination", f"{illum:.2f}", help="Uniformity score")
                else:
                    st.metric("Illumination", "N/A", help="Uniformity score")
            
            with col3:
                noise = img_meta.get('noise_level', 0)
                if noise is not None and not np.isnan(noise):
                    st.metric("Noise", f"{noise:.3f}", help="Noise level estimate")
                else:
                    st.metric("Noise", "N/A", help="Noise level estimate")
            
            with col4:
                # Empty column for grid alignment
                st.metric("", "")
            
            st.markdown("")  # spacing

            # Processing & Technical Metrics
            st.markdown("**Processing Performance**")
            proc_metrics = processor.processing_metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                proc_time = proc_metrics.get('processing_time', 0)
                if proc_time is not None:
                    st.metric("Time", f"{proc_time * 1000:.1f} ms")
                else:
                    st.metric("Time", "N/A")
            
            with col2:
                throughput = proc_metrics.get('processing_time_normalised', 0)
                if throughput is not None:
                    st.metric("Throughput", f"{throughput:.1f} MPx/s")
                else:
                    st.metric("Throughput", "N/A")
            
            with col3:
                size_x = img_meta.get('size_x')
                size_y = img_meta.get('size_y')
                if size_x is not None and size_y is not None:
                    st.metric("Resolution", f"{size_x} × {size_y}", help="Image dimensions in pixels")
                else:
                    st.metric("Resolution", "N/A", help="Image dimensions in pixels")
            
            with col4:
                if processor.calibration.get('um_per_pixel'):
                    st.metric("Calibration", f"{processor.calibration['um_per_pixel']:.2f} µm/px")
                else:
                    st.metric("Calibration", "None")


def main():
    st.title("Auto-Fluidics Dashboard")
    st.markdown("**Real-time droplet detection and analysis**")

    # Sidebar
    st.sidebar.header("Settings")

    # Detection parameters
    with st.sidebar.expander("Detection Parameters", expanded=False):
        st.markdown("**Hough Circle Transform Parameters**")
        
        use_custom = st.checkbox(
            "Override auto-detection",
            value=st.session_state.detection_config['use_custom'],
            help="Enable to manually tune detection parameters"
        )
        st.session_state.detection_config['use_custom'] = use_custom
        
        if use_custom:
            st.info("Adjust parameters and the image will automatically reprocess")
            
            col1, col2 = st.columns(2)
            
            with col1:
                dp = st.number_input(
                    "dp (resolution)",
                    min_value=1,
                    max_value=3,
                    value=st.session_state.detection_config['dp'],
                    help="Inverse ratio of accumulator resolution. 1 = same as input image"
                )
                st.session_state.detection_config['dp'] = dp
                
                param1 = st.slider(
                    "param1 (edge)",
                    min_value=10,
                    max_value=200,
                    value=st.session_state.detection_config['param1'],
                    help="Upper threshold for Canny edge detector. Lower = more circles detected"
                )
                st.session_state.detection_config['param1'] = param1
                
                min_radius = st.slider(
                    "Min Radius",
                    min_value=1,
                    max_value=100,
                    value=st.session_state.detection_config['minRadius'],
                    help="Minimum circle radius in pixels"
                )
                st.session_state.detection_config['minRadius'] = min_radius
            
            with col2:
                min_dist = st.slider(
                    "Min Distance",
                    min_value=10,
                    max_value=200,
                    value=st.session_state.detection_config['minDist'],
                    help="Minimum distance between circle centers"
                )
                st.session_state.detection_config['minDist'] = min_dist
                
                param2 = st.slider(
                    "param2 (threshold)",
                    min_value=10,
                    max_value=100,
                    value=st.session_state.detection_config['param2'],
                    help="Accumulator threshold. Lower = more circles detected"
                )
                st.session_state.detection_config['param2'] = param2
                
                max_radius = st.slider(
                    "Max Radius",
                    min_value=50,
                    max_value=1000,
                    value=st.session_state.detection_config['maxRadius'],
                    help="Maximum circle radius in pixels"
                )
                st.session_state.detection_config['maxRadius'] = max_radius
            
            if st.button("Reset to Defaults", use_container_width=True):
                st.session_state.detection_config = {
                    'dp': 1,
                    'minDist': 40,
                    'param1': 70,
                    'param2': 50,
                    'minRadius': 10,
                    'maxRadius': 500,
                    'use_custom': True
                }
                st.rerun()
        else:
            st.caption("Using automatic parameter detection based on image type")
            st.caption("Grayscale: param1=70, param2=50")
            st.caption("Blue channel: param1=40, param2=20")

    # Regime detection (optional)
    with st.sidebar.expander("Flow Classification", expanded=False):
        st.session_state.enable_regime_classification = st.checkbox(
            "Enable regime classification",
            value=st.session_state.enable_regime_classification,
            help="Classify flow regime (DRIPPING, JETTING, NO_FLOW)"
        )

        if st.session_state.enable_regime_classification:
            # Choose classifier type
            if ML_CLASSIFIER_AVAILABLE and st.session_state.ml_classifier is not None:
                classifier_type = st.radio(
                    "Classifier",
                    ["ML (CNN)", "Rule-based"],
                    index=0 if st.session_state.use_ml_classifier else 1,
                    help="ML classifier is more accurate if trained"
                )
                st.session_state.use_ml_classifier = (classifier_type == "ML (CNN)")
                
                if st.session_state.use_ml_classifier:
                    st.success("Using trained ML classifier")
                else:
                    st.info("Using rule-based heuristics")
            else:
                st.session_state.use_ml_classifier = False
                if not ML_CLASSIFIER_AVAILABLE:
                    st.warning("ML classifier unavailable. Install TensorFlow to enable.")
                elif st.session_state.ml_classifier is None:
                    st.warning("No trained model found. Using rule-based classifier.")
                    st.caption("Train a model with: python train_regime_classifier.py")
            
            # Rule-based parameters (only show if using rule-based)
            if not st.session_state.use_ml_classifier:
                st.markdown("**Rule-based Parameters**")
                min_particles = st.number_input("Min Particles (DRIPPING)", 1, 10, 3)
                aspect_threshold = st.slider("Aspect Ratio Threshold", 1.0, 5.0, 2.5, 0.1)
                cv_threshold = st.slider("CV Threshold", 0.0, 1.0, 0.3, 0.05)

                st.session_state.regime_detector.min_particles = min_particles
                st.session_state.regime_detector.aspect_ratio_threshold = aspect_threshold
                st.session_state.regime_detector.size_cv_threshold = cv_threshold

    # Calibration
    with st.sidebar.expander("Calibration", expanded=True):
        use_calibration = st.checkbox("Enable calibration", value=False)
        if use_calibration:
            um_per_pixel = st.number_input(
                "µm per pixel",
                0.0, 100.0, 2.0, 0.1,
                help="Determined by imaging a known reference object"
            )
            st.session_state.processor.set_calibration(um_per_pixel)
            st.success(f"Calibrated: {um_per_pixel} µm/px")
        else:
            st.session_state.processor.calibration['um_per_pixel'] = None

    st.sidebar.markdown("---")

    # Mode selection
    mode = st.sidebar.radio("Mode", ["Single Image", "Batch Analysis"])

    if mode == "Single Image":
        # Get list of sample images
        img_dir = Path("img")
        image_files = sorted(list(img_dir.glob("*.png")))

        if not image_files:
            st.error("No images found in img/ directory!")
            return

        # Image selector
        selected_image = st.sidebar.selectbox(
            "Select Image",
            image_files,
            format_func=lambda x: x.name,
            key='image_selector'
        )

        # Auto-process when image changes or config changes
        config_changed = st.session_state.last_config != st.session_state.detection_config
        if st.session_state.last_processed_image != selected_image or config_changed:
            st.session_state.last_processed_image = selected_image
            st.session_state.last_config = st.session_state.detection_config.copy()

            with st.spinner("Processing..."):
                process_and_display(
                    selected_image,
                    st.session_state.processor,
                    st.session_state.regime_detector,
                    show_regime=st.session_state.enable_regime_classification,
                    detection_config=st.session_state.detection_config,
                    ml_classifier=st.session_state.ml_classifier,
                    use_ml=st.session_state.use_ml_classifier
                )
        else:
            # Re-display current results
            process_and_display(
                selected_image,
                st.session_state.processor,
                st.session_state.regime_detector,
                show_regime=st.session_state.enable_regime_classification,
                detection_config=st.session_state.detection_config,
                ml_classifier=st.session_state.ml_classifier,
                use_ml=st.session_state.use_ml_classifier
            )

        # Show original in sidebar
        with st.sidebar:
            st.markdown("---")
            st.markdown("**Original Image**")
            orig_img = cv2.imread(str(selected_image))
            if orig_img is not None:
                orig_rgb = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
                st.image(orig_rgb, use_container_width=True)

    else:  # Batch Analysis
        st.markdown("### Batch Analysis")

        img_dir = Path("img")
        image_files = sorted(list(img_dir.glob("*.png")))

        if st.button("Analyze All Images", type="primary", use_container_width=True):
            progress_bar = st.progress(0)
            status_text = st.empty()

            results = []

            for i, img_path in enumerate(image_files):
                status_text.text(f"Processing {img_path.name}...")

                # Process
                st.session_state.processor.process_image(str(img_path))
                metrics = st.session_state.processor.particle_metrics

                regime_str = "N/A"
                confidence_str = "N/A"
                if st.session_state.enable_regime_classification:
                    regime, confidence, _ = st.session_state.regime_detector.detect_regime(metrics)
                    regime_str = regime_to_string(regime)
                    confidence_str = f"{confidence:.0%}"

                mean_size = metrics.get('mean_particle_size', 0)
                cv = metrics.get('coefficient_of_variation', 0)

                results.append({
                    'Image': img_path.name,
                    'Particles': metrics['num_particles'],
                    'Mean Size': f"{mean_size:.1f}" if mean_size and not np.isnan(mean_size) else "N/A",
                    'CV': f"{cv:.3f}" if cv and not np.isnan(cv) else "N/A",
                    'Regime': regime_str,
                    'Confidence': confidence_str
                })

                progress_bar.progress((i + 1) / len(image_files))

            status_text.success("Processing complete!")
            progress_bar.empty()

            # Display results table
            import pandas as pd
            df = pd.DataFrame(results)

            st.markdown("---")
            st.dataframe(df, use_container_width=True, hide_index=True)

            # Summary statistics
            st.markdown("---")
            st.markdown("### Summary")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Total Images", len(results))

            with col2:
                avg_particles = np.mean([r['Particles'] for r in results])
                st.metric("Avg Particles", f"{avg_particles:.1f}")

            with col3:
                detected = sum(1 for r in results if r['Particles'] > 0)
                st.metric("Images w/ Particles", detected)

            with col4:
                if st.session_state.enable_regime_classification:
                    dripping = sum(1 for r in results if r['Regime'] == 'DRIPPING')
                    st.metric("DRIPPING", dripping)

    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Auto-Fluidics v1.0**")
    st.sidebar.caption("Microfluidic droplet detection")
    st.sidebar.caption("[Documentation](https://github.com/FinOrr/auto-fluidics)")


if __name__ == "__main__":
    main()
