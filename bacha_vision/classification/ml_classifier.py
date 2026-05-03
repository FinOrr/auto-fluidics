"""
ML-based Regime Classifier for Microfluidics

Uses a lightweight CNN (MobileNetV2) to classify flow regimes from images.
Operates on channel region crops for robust classification.

Classes:
- DRIPPING: Round droplets visible in channel (desired state)
- JETTING: Continuous stream, no discrete droplets (unstable)
- CO_FLOW: Co-flowing streams without droplet formation
- NO_FLOW: Empty channel, no visible flow

Author: Auto-Fluidics Project
"""

import numpy as np
import cv2
from pathlib import Path
from typing import Tuple, Optional
import pickle

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
    from tensorflow.keras.models import Model
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

from enum import Enum

# Import FlowRegime from regime_detector to ensure consistency
try:
    from bacha_vision.classification.regime_detector import FlowRegime
except ImportError:
    # Fallback definition if import fails
    class FlowRegime(Enum):
        """Flow regime classifications"""
        NO_FLOW = 0
        DRIPPING = 1
        JETTING = 2
        CO_FLOW = 3
        UNKNOWN = 4


class MLRegimeClassifier:
    """
    ML-based flow regime classifier using MobileNetV2.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the ML classifier.
        
        Args:
            model_path: Path to saved model weights. If None, creates untrained model.
        """
        if not TF_AVAILABLE:
            raise ImportError(
                "TensorFlow not available. Install with: pip install tensorflow"
            )
        
        self.img_size = (224, 224)  # MobileNetV2 input size
        self.num_classes = 4  # NO_FLOW, DRIPPING, JETTING, CO_FLOW
        self.class_names = ['NO_FLOW', 'DRIPPING', 'JETTING', 'CO_FLOW']
        
        # Build model
        self.model = self._build_model()
        
        # Load weights if provided
        if model_path and Path(model_path).exists():
            self.load_model(model_path)
            self.trained = True
        else:
            self.trained = False
    
    def _build_model(self) -> Model:
        """
        Build MobileNetV2-based classifier.
        
        Returns:
            Keras model
        """
        # Load pre-trained MobileNetV2 (ImageNet weights)
        base_model = MobileNetV2(
            input_shape=(*self.img_size, 3),
            include_top=False,
            weights='imagenet'
        )
        
        # Freeze base model layers (transfer learning)
        base_model.trainable = False
        
        # Add classification head
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dropout(0.2)(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.2)(x)
        predictions = Dense(self.num_classes, activation='softmax')(x)
        
        # Create model
        model = Model(inputs=base_model.input, outputs=predictions)
        
        # Compile
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for model input.
        
        Args:
            image: Input image (BGR format from OpenCV)
            
        Returns:
            Preprocessed image ready for model
        """
        # Convert BGR to RGB
        if len(image.shape) == 2:
            # Grayscale - convert to RGB
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize to model input size
        image = cv2.resize(image, self.img_size)
        
        # Normalize to [-1, 1] (MobileNetV2 preprocessing)
        image = image.astype(np.float32)
        image = (image / 127.5) - 1.0
        
        return image
    
    def predict(self, image: np.ndarray, channel_roi: Optional[Tuple] = None) -> Tuple[FlowRegime, float, np.ndarray]:
        """
        Predict flow regime from image.
        
        Args:
            image: Input image (BGR format from OpenCV)
            channel_roi: Optional (x, y, w, h) to crop channel region
            
        Returns:
            Tuple of (regime, confidence, class_probabilities)
        """
        if not self.trained:
            return FlowRegime.UNKNOWN, 0.0, np.zeros(self.num_classes)
        
        # Extract channel region if ROI provided
        if channel_roi is not None:
            x, y, w, h = channel_roi
            image = image[y:y+h, x:x+w]
        
        # Preprocess
        processed = self.preprocess_image(image)
        processed = np.expand_dims(processed, axis=0)  # Add batch dimension
        
        # Predict
        probs = self.model.predict(processed, verbose=0)[0]
        
        # Get predicted class
        class_idx = np.argmax(probs)
        confidence = probs[class_idx]
        
        # Map to FlowRegime enum
        regime_map = {
            0: FlowRegime.NO_FLOW,
            1: FlowRegime.DRIPPING,
            2: FlowRegime.JETTING,
            3: FlowRegime.CO_FLOW
        }
        regime = regime_map.get(class_idx, FlowRegime.UNKNOWN)
        
        return regime, confidence, probs
    
    def train(self, train_data_dir: str, validation_split: float = 0.2, 
              epochs: int = 20, batch_size: int = 32):
        """
        Train the classifier on labeled data.
        
        Expected directory structure:
        train_data_dir/
            NO_FLOW/
                image1.png
                image2.png
                ...
            DRIPPING/
                image1.png
                image2.png
                ...
            JETTING/
                image1.png
                image2.png
                ...
            CO_FLOW/
                image1.png
                image2.png
                ...
        
        Args:
            train_data_dir: Path to training data directory
            validation_split: Fraction of data to use for validation
            epochs: Number of training epochs
            batch_size: Training batch size
            
        Returns:
            Training history
        """
        # Data augmentation for training
        train_datagen = keras.preprocessing.image.ImageDataGenerator(
            preprocessing_function=lambda x: (x / 127.5) - 1.0,
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            validation_split=validation_split
        )
        
        # Training generator
        train_generator = train_datagen.flow_from_directory(
            train_data_dir,
            target_size=self.img_size,
            batch_size=batch_size,
            class_mode='categorical',
            subset='training',
            classes=self.class_names,
            color_mode='rgb'
        )
        
        # Validation generator
        val_generator = train_datagen.flow_from_directory(
            train_data_dir,
            target_size=self.img_size,
            batch_size=batch_size,
            class_mode='categorical',
            subset='validation',
            classes=self.class_names,
            color_mode='rgb'
        )
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-6
            )
        ]
        
        # Train
        history = self.model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=epochs,
            callbacks=callbacks
        )
        
        self.trained = True
        return history
    
    def save_model(self, path: str):
        """Save model weights."""
        self.model.save_weights(path)
        print(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load model weights."""
        self.model.load_weights(path)
        self.trained = True
        print(f"Model loaded from {path}")
    
    def evaluate(self, test_data_dir: str, batch_size: int = 32):
        """
        Evaluate model on test data.
        
        Args:
            test_data_dir: Path to test data (same structure as training)
            batch_size: Evaluation batch size
            
        Returns:
            Test loss and accuracy
        """
        test_datagen = keras.preprocessing.image.ImageDataGenerator(
            preprocessing_function=lambda x: (x / 127.5) - 1.0
        )
        
        test_generator = test_datagen.flow_from_directory(
            test_data_dir,
            target_size=self.img_size,
            batch_size=batch_size,
            class_mode='categorical',
            classes=self.class_names,
            shuffle=False
        )
        
        results = self.model.evaluate(test_generator)
        return dict(zip(self.model.metrics_names, results))


# regime_to_string and is_safe_regime are now imported from regime_detector
# to maintain a single source of truth for regime handling
