# Pneumonia Detection App 🫁

An AI-powered chest X-ray analysis tool that predicts pneumonia and provides detailed explanations of the findings. **Ready for Streamlit Cloud deployment!**


## Features

### 🔬 Core Functionality
- **AI-Powered Detection**: Uses a pre-trained DenseNet121 model for pneumonia detection
- **Confidence Scoring**: Provides prediction confidence levels and probability breakdown
- **Medical-Grade Interface**: Clean, professional interface suitable for educational use

### 🤖 AI Explanations
- **Natural Language Explanations**: Detailed explanations of findings in plain English
- **Clinical Context**: Appropriate next steps and medical recommendations
- **Risk Assessment**: Confidence-based warnings and guidance

### ☁️ Cloud-Ready Features
- **Streamlit Cloud Compatible**: Optimized for easy cloud deployment
- **Environment Variable Support**: Works with both local `.env` and Streamlit secrets
- **Relative File Paths**: No hardcoded paths, works anywhere
- **Optimized Dependencies**: Cloud-compatible package versions

## Quick Start

### 🚀 Deploy on Streamlit Cloud (Recommended)

1. **Fork this repository**
2. **Go to [Streamlit Cloud](https://share.streamlit.io)**
3. **Create new app** pointing to your forked repo
4. **Add secrets** in Streamlit Cloud dashboard:
   ```toml
   GROQ_API_KEY = "your_groq_api_key"
   HF_TOKEN = "your_huggingface_token"
   ```
5. **Deploy!** 🎉

### 💻 Run Locally

```bash
# Clone repository
git clone https://github.com/anshKjha10/pneumonia-detector.git
cd pneumonia-detector

# Install dependencies
pip install -r requirements.txt

# Set environment variables
# Create .env file with:
# GROQ_API_KEY=your_key_here
# HF_TOKEN=your_token_here

# Run app
streamlit run app.py
```

## Technical Implementation

### Model Architecture
- **Base Model**: DenseNet121 pre-trained on ImageNet
- **Input Size**: 256x256 RGB images
- **Output**: Binary classification (Normal/Pneumonia)
- **Framework**: TensorFlow 2.16.1
- **Model Size**: ~8MB (GitHub compatible)

### Dependencies
- **TensorFlow 2.16.1**: For model inference
- **Streamlit**: Web interface
- **LangChain + GROQ**: AI explanations
- **OpenCV**: Image processing
- **PIL/Pillow**: Image handling

## Usage

1. **Upload X-ray Image** → PNG, JPG, or JPEG format
2. **Click "Start Analysis"** → AI processes the image
3. **View Results** → Prediction, confidence, and probability breakdown
4. **Read AI Explanation** → Detailed natural language explanation

## Medical Disclaimer ⚠️

**IMPORTANT**: This tool is for educational and research purposes only. It is NOT intended for:
- Medical diagnosis
- Treatment decisions
- Clinical decision-making
- Replacing professional medical consultation

Always consult qualified healthcare professionals for medical advice and interpretation of medical images.

## API Keys Required

### GROQ API Key
- Get free API key from [GROQ](https://console.groq.com)
- Used for AI explanations
- Add to Streamlit secrets or `.env` file

### HuggingFace Token (Optional)
- Get from [HuggingFace](https://huggingface.co/settings/tokens)
- Used for some embedding models
- Add to Streamlit secrets or `.env` file

## File Structure

```
pneumonia-detector/
├── 📄 app.py                    # Main Streamlit application
├── 🧠 pneumonia_detection_model.h5  # Pre-trained model
├── 📊 model_info.json           # Model metadata
├── 📋 requirements.txt          # Python dependencies
├── 📖 README.md                 # This file
├── 🚀 DEPLOYMENT.md             # Detailed deployment guide
├── 🚫 .gitignore               # Git ignore rules
└── ⚙️ .streamlit/
    ├── config.toml             # Streamlit configuration
    └── secrets.toml            # Secrets template
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test locally
5. Submit a pull request

## License

This project is for educational purposes. Please ensure compliance with medical software regulations in your jurisdiction.

---

**🎯 Ready to deploy? Check out [DEPLOYMENT.md](DEPLOYMENT.md) for detailed instructions!**
