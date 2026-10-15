---
title: Aerial Scene Classifier
emoji: 🛰️
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 5.50.0
python_version: "3.11"
app_file: app.py
pinned: false
license: mit
short_description: Classify satellite images into 45 land-use scenes
---

# Aerial Scene Classifier

Upload an aerial or satellite image and the model predicts which of 45 land-use scene
categories it shows (airport, harbor, farmland, residential areas, and more).

- **Model:** ResNet-50 fine-tuned on NWPU-RESISC45, with 94.4% test accuracy
- **Code, training and evaluation:** https://github.com/esammostafa9-cloud/resisc45-scene-classification

The example images come from the held-out test split of NWPU-RESISC45 (Cheng et al., 2017).
