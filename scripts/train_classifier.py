#!/usr/bin/env python3
"""
Training script for ML-based regime classifier.

Usage:
    1. Organize your labeled images into:
       training_data/
           NO_FLOW/
               *.png
           DRIPPING/
               *.png
           JETTING/
               *.png
           CO_FLOW/
               *.png
    
    2. Run: python train_regime_classifier.py --data_dir training_data --epochs 20
    
    3. Model will be saved to: models/regime_classifier_weights.h5

Author: Auto-Fluidics Project
"""

import argparse
from pathlib import Path
import matplotlib.pyplot as plt

from bacha_vision.classification.ml_classifier import MLRegimeClassifier


def plot_training_history(history, save_path='training_history.png'):
    """Plot training and validation metrics."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Accuracy
    ax1.plot(history.history['accuracy'], label='Train')
    ax1.plot(history.history['val_accuracy'], label='Validation')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # Loss
    ax2.plot(history.history['loss'], label='Train')
    ax2.plot(history.history['val_loss'], label='Validation')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('Model Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Training history saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Train regime classifier')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to training data directory')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Training batch size')
    parser.add_argument('--validation_split', type=float, default=0.2,
                        help='Fraction of data for validation')
    parser.add_argument('--output_dir', type=str, default='models',
                        help='Directory to save trained model')
    
    args = parser.parse_args()
    
    # Validate data directory
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise ValueError(f"Data directory not found: {data_dir}")
    
    # Check for required subdirectories
    required_classes = ['NO_FLOW', 'DRIPPING', 'JETTING', 'CO_FLOW']
    for class_name in required_classes:
        class_dir = data_dir / class_name
        if not class_dir.exists():
            raise ValueError(f"Missing class directory: {class_dir}")
        
        num_images = len(list(class_dir.glob('*.png'))) + len(list(class_dir.glob('*.jpg')))
        print(f"Found {num_images} images in {class_name}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Initialize classifier
    print("\nInitializing classifier...")
    classifier = MLRegimeClassifier()
    
    # Train
    print("\nStarting training...")
    history = classifier.train(
        train_data_dir=str(data_dir),
        validation_split=args.validation_split,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
    
    # Save model
    model_path = output_dir / 'regime_classifier.weights.h5'
    classifier.save_model(str(model_path))
    
    # Plot training history
    plot_training_history(history, save_path=str(output_dir / 'training_history.png'))
    
    print("\n" + "="*50)
    print("Training complete!")
    print(f"Model saved to: {model_path}")
    print(f"Final validation accuracy: {history.history['val_accuracy'][-1]:.3f}")
    print("="*50)


if __name__ == '__main__':
    main()
