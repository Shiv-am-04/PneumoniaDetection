import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import preprocess_input
import numpy as np

def prediction(path,model):
    img = image.load_img(path,target_size=(224,224))
    img = image.img_to_array(img)
    img = np.expand_dims(img,axis=0)

    img_data = preprocess_input(img)

    pred = model.predict(img_data)

    class_label = (pred > 0.5).astype("int32")

    return class_label

model = load_model('new_best_model.keras')

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