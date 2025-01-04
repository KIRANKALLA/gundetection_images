%%writefile ody8.py
import streamlit as st
import numpy as np
from ultralytics import YOLO
from PIL import Image
from skimage.io import imread

model = YOLO('wd.pt')
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    #image = np.array(Image.open(uploaded_file))
    image = imread(uploaded_file)
    results = model(image)
    annotated_image = results[0].plot()
    st.image(annotated_image, caption="Processed Image with Detections")
