# Streamlit Cloud Deployment Guide 🚀

## Prerequisites
1. GitHub account
2. Streamlit Cloud account (https://share.streamlit.io)
3. GROQ API key
4. HuggingFace token (optional)

## Deployment Steps

### 1. Push to GitHub
The code is ready for GitHub. Run these commands:

```bash
# Add remote repository (replace with your GitHub repo URL)
git remote add origin https://github.com/anshKjha10/pneumonia-detector.git

# Push to GitHub
git branch -M main
git push -u origin main
```

### 2. Deploy on Streamlit Cloud

1. **Go to Streamlit Cloud**: https://share.streamlit.io
2. **Sign in** with your GitHub account
3. **Click "New app"**
4. **Select repository**: `anshKjha10/pneumonia-detector`
5. **Set main file path**: `app.py`
6. **Configure secrets**:
   - Go to "Advanced settings" → "Secrets"
   - Add your secrets in TOML format:
   ```toml
   GROQ_API_KEY = "your_actual_groq_api_key"
   HF_TOKEN = "your_actual_huggingface_token"
   ```
7. **Click "Deploy"**

### 3. Monitor Deployment
- Watch the build logs for any errors
- The app will be available at your custom Streamlit URL
- Initial deployment may take 5-10 minutes due to model download

## Important Notes

### ✅ Cloud-Ready Features Added:
- **Relative file paths**: Model loading works in cloud environment
- **Environment variable handling**: Supports both local `.env` and Streamlit secrets
- **Optimized dependencies**: Uses `opencv-python-headless` for cloud compatibility
- **Error handling**: Graceful fallbacks for missing dependencies

### 📋 File Structure for Deployment:
```
Pneumonia-Detector/
├── app.py                          # Main Streamlit application
├── pneumonia_detection_model.h5    # Pre-trained model (8MB)
├── model_info.json                 # Model metadata
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
├── DEPLOYMENT.md                   # This deployment guide
├── .gitignore                      # Git ignore rules
└── .streamlit/
    ├── config.toml                 # Streamlit configuration
    └── secrets.toml                # Secrets template (not pushed to GitHub)
```

### 🔑 Required Environment Variables:
- `GROQ_API_KEY`: Your GROQ API key for AI explanations
- `HF_TOKEN`: HuggingFace token (optional, for some models)

### 🚨 Troubleshooting:

**If deployment fails:**
1. Check build logs in Streamlit Cloud dashboard
2. Verify all requirements are in requirements.txt
3. Ensure secrets are properly configured
4. Check file paths are relative, not absolute

**Common issues:**
- **Model loading errors**: Ensure model file is included in repository
- **API key errors**: Verify secrets are correctly set in Streamlit Cloud
- **Memory issues**: The model is ~8MB, should work fine on Streamlit Cloud

## Success Checklist ✅

Before deploying, ensure:
- [ ] All file paths are relative
- [ ] Environment variables use both os.environ and st.secrets
- [ ] requirements.txt includes all dependencies
- [ ] .gitignore excludes sensitive files
- [ ] Model file is included in repository
- [ ] README.md is complete
- [ ] Git repository is clean and committed

## Post-Deployment

Once deployed successfully:
1. Test the app with sample X-ray images
2. Verify AI explanations are working
3. Check model predictions are accurate
4. Share the URL with users

Your app will be accessible at: `https://your-app-name.streamlit.app`

## Support

If you encounter issues:
- Check Streamlit Cloud documentation
- Review build logs for specific errors
- Ensure all dependencies are compatible
- Verify model file integrity
