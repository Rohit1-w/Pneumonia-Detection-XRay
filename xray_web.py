import streamlit as st
import tensorflow as tf
import cv2
from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt
import pyttsx3
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Define class names
class_names = ["Normal", "Pneumonia"]

@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model("model/xray_model.hdf5")
    return model

with st.spinner('Model is being loaded..'):
    model = load_model()

st.write("""
         # Pneumonia Identification System
         """)

file = st.file_uploader("Please upload a chest scan file", type=["jpg", "jpeg", "png"])

st.set_option('deprecation.showfileUploaderEncoding', False)

def import_and_predict(image_data, model):
    size = (180, 180)
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    image = np.asarray(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image / 255.0  # Normalize the image
    img_reshape = np.expand_dims(image, axis=0)
    prediction = model.predict(img_reshape)
    return prediction

def get_treatment_recommendation(predicted_class):
    if predicted_class == "Normal":
        return "✅ No signs of pneumonia detected. Maintain a healthy lifestyle and regular checkups."
    else:
        return "⚠️ Pneumonia detected! Recommended to consult a doctor immediately. Treatment may include antibiotics, antiviral medications, rest, and hydration. Hospitalization may be needed in severe cases."

def text_to_speech(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    predictions = import_and_predict(image, model)
    score = tf.nn.softmax(predictions[0]).numpy()
    
    predicted_class = class_names[np.argmax(score)]
    confidence = 100 * np.max(score)
    
    st.write(f"This image most likely belongs to **{predicted_class}** with a confidence of **{confidence:.2f}%**")
    
    # Display treatment recommendation
    treatment = get_treatment_recommendation(predicted_class)
    st.write(treatment)
    
    # Add a voice button to read the treatment details
    if st.button("🔊 Voice"):
        text_to_speech(treatment)
    
    # Plot the prediction result
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].bar(class_names, score, color=['blue', 'red'])
    ax[0].set_ylabel("Confidence (%)")
    ax[0].set_title("Prediction Results")
    
    # Prediction dot diagram
    ax[1].scatter(class_names, score, color=['blue', 'red'], marker='o', s=100)
    ax[1].set_ylabel("Confidence (%)")
    ax[1].set_title("Prediction Dot Diagram")
    
    st.pyplot(fig)
    
    # Generate a sample confusion matrix (for demonstration purposes, using synthetic data)
    y_true = np.random.choice([0, 1], size=10)  # Example ground truth labels
    y_pred = np.random.choice([0, 1], size=10)  # Example prediction labels
    cm = confusion_matrix(y_true, y_pred)
    
    fig_cm, ax_cm = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, ax=ax_cm)
    ax_cm.set_xlabel("Predicted")
    ax_cm.set_ylabel("Actual")
    ax_cm.set_title("Confusion Matrix")
    st.pyplot(fig_cm)
