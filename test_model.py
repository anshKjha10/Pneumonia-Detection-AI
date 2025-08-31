import tensorflow as tf
import os
import json

print('TensorFlow version:', tf.__version__)

model_path = r'D:\My Projects\Pneumonia-Detector\pneumonia_detection_model.h5'
if os.path.exists(model_path):
    try:
        # Try loading with different methods
        print("Attempting to load model...")
        
        # Method 1: Load without compilation
        try:
            model = tf.keras.models.load_model(model_path, compile=False)
            print('Model loaded successfully with compile=False!')
            print('Input shape:', model.input_shape)
            print('Output shape:', model.output_shape)
        except Exception as e1:
            print(f'Method 1 failed: {e1}')
            
            # Method 2: Try loading with custom objects
            try:
                print("Trying method 2 with custom objects...")
                
                # Create a custom InputLayer that handles batch_shape
                class CompatibleInputLayer(tf.keras.layers.InputLayer):
                    def __init__(self, batch_shape=None, input_shape=None, **kwargs):
                        if batch_shape is not None and input_shape is None:
                            input_shape = batch_shape[1:]  # Remove batch dimension
                        super().__init__(input_shape=input_shape, **kwargs)
                
                model = tf.keras.models.load_model(
                    model_path, 
                    custom_objects={'InputLayer': CompatibleInputLayer},
                    compile=False
                )
                print('Model loaded successfully with custom objects!')
                print('Input shape:', model.input_shape)
                print('Output shape:', model.output_shape)
                
            except Exception as e2:
                print(f'Method 2 failed: {e2}')
                
                # Method 3: Try loading weights only
                try:
                    print("Trying method 3 - recreating model architecture...")
                    
                    # Recreate the model architecture based on the info
                    model = tf.keras.Sequential([
                        tf.keras.layers.Input(shape=(256, 256, 3)),
                        tf.keras.applications.DenseNet121(
                            weights='imagenet',
                            include_top=False,
                            input_shape=(256, 256, 3)
                        ),
                        tf.keras.layers.GlobalAveragePooling2D(),
                        tf.keras.layers.Dense(128, activation='relu'),
                        tf.keras.layers.Dropout(0.5),
                        tf.keras.layers.Dense(2, activation='softmax')
                    ])
                    
                    # Try to load weights
                    try:
                        model.load_weights(model_path)
                        print('Weights loaded successfully!')
                        print('Input shape:', model.input_shape)
                        print('Output shape:', model.output_shape)
                    except:
                        print('Could not load weights - model architecture might be different')
                        
                except Exception as e3:
                    print(f'Method 3 failed: {e3}')
                    print('All methods failed')
        
    except Exception as e:
        print(f'Error loading model: {e}')
else:
    print('Model file not found!')
