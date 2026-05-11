# 🚀 Deployment Guide

## Method 1: Hugging Face Spaces (Easiest - FREE)

1. **Create Account**: https://huggingface.co/join
2. **New Space**: Click "New Space"
3. **Settings**:
   - Name: `toxic-detector`
   - SDK: `Streamlit`
   - Hardware: `CPU Basic`
4. **Upload Files**:
   - app.py
   - requirements.txt
   - README.md
5. **Wait 2 minutes** for deployment
6. **Your URL**: `https://huggingface.co/spaces/YOUR_USERNAME/toxic-detector`

## Method 2: Streamlit Cloud (FREE)

1. **Create GitHub Repo** (public)
2. **Upload** all files
3. **Go to**: https://share.streamlit.io
4. **Connect** GitHub
5. **Deploy**

## Method 3: Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
