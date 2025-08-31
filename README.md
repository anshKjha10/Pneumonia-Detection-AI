# Pneumonia Detection App with AI Explanation 🫁

An AI-powered chest X-ray analysis tool that predicts pneumonia and provides detailed explanations of the findings.

## Features

### 🔬 Core Functionality
- **AI-Powered Detection**: Uses a pre-trained DenseNet121 model for pneumonia detection
- **Confidence Scoring**: Provides prediction confidence levels and probability breakdown
- **Medical-Grade Interface**: Clean, professional interface suitable for educational use

### 🤖 AI Explanations
- **Natural Language Explanations**: Detailed explanations of findings in plain English
- **Clinical Context**: Appropriate next steps and medical recommendations
- **Risk Assessment**: Confidence-based warnings and guidance

## Technical Implementation

### Model Architecture
- **Base Model**: DenseNet121 pre-trained on ImageNet
- **Input Size**: 256x256 RGB images
- **Output**: Binary classification (Normal/Pneumonia)
- **Framework**: TensorFlow 2.16.1

## Requirements

```txt
tensorflow==2.16.1
numpy<2
streamlit
pandas
langchain-groq
langchain-huggingface
opencv-python
pillow
matplotlib
python-dotenv
```

## Usage

1. **Start the Application**
   ```bash
   streamlit run app.py
   ```

2. **Upload X-ray Image**
   - Support formats: PNG, JPG, JPEG
   - Recommended: Clear, front-facing chest X-rays

3. **Analyze Results**
   - View AI prediction and confidence
   - Read detailed AI explanation

## Medical Disclaimer ⚠️

**IMPORTANT**: This tool is for educational and research purposes only. It is NOT intended for:
- Medical diagnosis
- Treatment decisions
- Clinical decision-making
- Replacing professional medical consultation

Always consult qualified healthcare professionals for medical advice and interpretation of medical images.

---

*This application demonstrates AI-assisted medical image analysis for educational purposes.*
