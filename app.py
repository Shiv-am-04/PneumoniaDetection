import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import preprocess_input
import numpy as np
import requests
import os

model_url = 'https://github.com/Shiv-am-04/PneumoniaDetection/releases/download/v.0.1/new_best_model.keras'
model_path = 'model.keras'

if not os.path.exists(model_path):
    response = requests.get(model_url)
    with open(model_path,'wb') as f:
        f.write(response.content)

model = load_model(model_path)

def prediction(path,model):
    img = image.load_img(path,target_size=(224,224))
    img = image.img_to_array(img)
    img = np.expand_dims(img,axis=0)

    img_data = preprocess_input(img)

    pred = model.predict(img_data)

    class_label = (pred > 0.5).astype("int32")

    return class_label

# streamlit app

st.set_page_config("Pneumonia Detection",page_icon='lungs')

st.title(':blue[**Check Your Lungs**] :stethoscope:')

xray = st.file_uploader(label=':red[**Upload Your Xray**]',accept_multiple_files=False)

if xray:
    class_label = prediction(xray,model=model)

    if class_label == 0:
        st.success("your lungs looks normal, you don't have neumonia, still consult your doctor")
    else:
        st.warning("It looks like you have Neumonia.Don't worry ,do chekup as soon as possible you will get cured")
