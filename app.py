import streamlit as st
import pandas as pd
import os
from langchain_groq import ChatGroq
import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
from PIL import Image

# Suppress TensorFlow warnings for better UX
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
tf.get_logger().setLevel('ERROR')

from dotenv import load_dotenv
load_dotenv()

# Handle environment variables for both local and cloud deployment
try:
    groq_api_key = os.environ.get('GROQ_API_KEY') or st.secrets.get("GROQ_API_KEY")
except:
    groq_api_key = os.environ.get('GROQ_API_KEY', '')

if not groq_api_key:
    st.error("GROQ API key not found. Please set GROQ_API_KEY in your environment or secrets.")
    st.stop()

llm = ChatGroq(model = 'gemma2-9b-it', api_key=groq_api_key)

st.set_page_config(
    page_title="Pneumonia Detection with AI Explanation",
    page_icon="🫁",
    layout="wide"
    )
st.title("Pneumonia Detection with AI Explanation 🫁")

st.sidebar.markdown('''## About this App 📖

**🫁 Pneumonia Detection Tool**

This AI-powered application uses advanced deep learning to analyze chest X-ray images and detect potential signs of pneumonia.

**🔬 How it works:**
- Upload a chest X-ray image
- AI analyzes the image using DenseNet121 model
- Get prediction with confidence score
- Receive detailed AI explanation

**🎯 Key Features:**
- Real-time X-ray analysis
- Confidence-based predictions
- Natural language explanations
- Educational insights

**⚠️ Important:** 
This is for educational purposes only - always consult healthcare professionals for medical advice.
           ''')

# load model
@st.cache_resource
def load_model():
    # Use relative path that works both locally and in cloud deployment
    model_path = os.path.join(os.path.dirname(__file__), 'pneumonia_detection_model.h5')
    
    if os.path.exists(model_path):
        try:
            loaded_model = keras.models.load_model(model_path, compile=False)
            st.success("Model loaded successfully!")
            return loaded_model
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None
    else:
        st.error(f"Model file not found at: {model_path}")
        return None
        
def preprocess_image(image):
    try:
        if isinstance(image, Image.Image):
            image = np.array(image)
        elif hasattr(image, 'read'): 
            image = Image.open(image)
            image = np.array(image)
        
        if len(image.shape) == 3 and image.shape[2] == 4: 
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        elif len(image.shape) == 3 and image.shape[2] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif len(image.shape) == 2:  
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            
        image = cv2.resize(image, (256, 256))
        image = image.astype(np.float32) / 255.0
        image = np.expand_dims(image, axis=0)
        return image

    except Exception as e:
        st.error(f"Error preprocessing image: {e}")
        return None
    
def predict_pneumonia(model, processed_image):
    try:
        prediction = model.predict(processed_image)
        class_names = ['NORMAL', 'PNEUMONIA']
        
        predicted_class = np.argmax(prediction[0])
        confidence = np.max(prediction[0])
   
        return {
            'prediction': class_names[predicted_class],
            'confidence': float(confidence),
            'probabilities': {
                'NORMAL': float(prediction[0][0]),
                'PNEUMONIA': float(prediction[0][1])
            }
        }
        
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        return None
    
def generate_explanation(prediction, llm):
    prediction_result = prediction
    confidence = prediction_result['confidence']
    if confidence < 0.6:
        st.warning("⚠️ Low confidence in prediction. Please consult a healthcare professional for accurate diagnosis.")
    try:
        prompt = f"""
        You are a medical AI assistant helping to explain chest X-ray analysis results to patients and healthcare providers.
        
        An AI model has analyzed a chest X-ray and provided these results:
        - Prediction: {prediction_result['prediction']}
        - Confidence: {prediction_result['confidence']:.2%}
        - Normal probability: {prediction_result['probabilities']['NORMAL']:.2%}
        - Pneumonia probability: {prediction_result['probabilities']['PNEUMONIA']:.2%}
        
        Provide a comprehensive explanation including:
        1. What the prediction means in simple terms
        2. Interpretation of the confidence level and what it suggests
        3. What these probabilities indicate about the X-ray findings
        4. Important medical disclaimers and limitations
        5. Appropriate next steps and recommendations
        6. When to seek immediate vs routine medical attention
        
        Keep explanations clear for non-medical users while maintaining accuracy.
        Use clear headings, bullet points, and avoid overly technical jargon.
        Always emphasize that this is AI-assisted analysis, not a medical diagnosis.
        """
        
        response = llm.invoke(prompt)
        return response.content
    
    except Exception as e:
        st.error(f"Error generating explanation: {e}")
        return "Error generating explanation."
    
def main():
    model = load_model()
    if model is None:
        st.error("Failed to load the pneumonia detection model.")
        return
    
    image = st.file_uploader(
        "Upload a chest X-ray image", 
        type=['png', 'jpg', 'jpeg'],
        help = "Upload a clear chest X-ray image in PNG, JPG or JPEG format."
        )
    
    if st.button("Start Analysis"):

        if image is not None:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Uploaded X-ray Image")
                display_image = Image.open(image)
                st.image(display_image, caption="Uploaded X-ray Image", width="stretch")
                
                st.write(f"**Image size:** {display_image.size}")
                st.write(f"**Image mode:** {display_image.mode}")
                
            with col2:
                st.subheader("🤖 AI Analysis")
                
                with st.spinner("Analyzing X-ray..."):
                    # Reset file pointer for processing
                    image.seek(0)
                    processed_image = preprocess_image(image)
                    
                    if processed_image is not None:
                        prediction_result = predict_pneumonia(model, processed_image)
                        
                        if prediction_result:
                            prediction = prediction_result['prediction']
                            confidence = prediction_result['confidence']
                            
                            if prediction == 'NORMAL':
                                st.success(f"**Prediction: {prediction}**")
                            else:
                                st.warning(f"**Prediction: {prediction}**")
                            
                            st.write(f"**Confidence:** {confidence:.1%}")
                            
                            st.write("**Probability Breakdown:**")
                            prob_normal = prediction_result['probabilities']['NORMAL']
                            prob_pneumonia = prediction_result['probabilities']['PNEUMONIA']
                            
                            st.progress(prob_normal, text=f"Normal: {prob_normal:.1%}")
                            st.progress(prob_pneumonia, text=f"Pneumonia: {prob_pneumonia:.1%}")
                            
            st.markdown("---")
            st.subheader("🩺 AI Explanation")
            
            with st.spinner("Generating explanation..."):
                explanation = generate_explanation(prediction_result, llm)
                st.markdown(explanation)
                
            st.markdown("---")
            st.error("""
            **⚠️ IMPORTANT MEDICAL DISCLAIMER:**
            This tool is for educational and research purposes only. It is NOT intended for medical diagnosis, treatment, or clinical decision-making. Always consult qualified healthcare professionals for medical advice.
            """)
            

if __name__ == "__main__":
    main()