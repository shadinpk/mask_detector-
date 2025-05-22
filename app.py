import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Load your trained model
model = load_model('mask_detector.h5')

# Set the image size expected by your model
IMG_SIZE = 224  # You had 244, but many models like MobileNetV2 use 224. Adjust if your model uses something else.

st.title("😷 Mask Detector App")

st.markdown("Upload an image to check whether the person is **wearing a mask** or **not**.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Open and display the uploaded image
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Preprocess the image
        img = image.resize((IMG_SIZE, IMG_SIZE))
        img_array = np.array(img) / 255.0  # normalize pixel values
        img_array = np.expand_dims(img_array, axis=0)  # add batch dimension

        # Make prediction
        prediction = model.predict(img_array)[0][0]

        # Output prediction
        if prediction < 0.5:
            st.success("Prediction: **Without Mask**")
            st.write(f"**Confidence:** {(1 - prediction) * 100:.2f}%")
        else:
            st.error("Prediction: **With Mask**")
            st.write(f"**Confidence:** {prediction * 100:.2f}%")

    except Exception as e:
        st.error("There was an error processing the image.")
        st.exception(e)
else:
    st.info("Please upload an image file.")
