import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import pandas as pd

model=load_model('capstone.h5')

meta_df = pd.read_csv('archive/Meta.csv')

class_names = dict(zip(meta_df['ClassId'], meta_df['ClassId']))

def process_image(img):  # Burada fonksiyon adını doğru yazalım
    img = img.resize((32, 32)) 
    img = np.array(img)
    img = img / 255.0  
    img = np.expand_dims(img, axis=0)
    return img

st.title('Trafik işareti Sınıflandırma Uygulaması 🚦')
st.write("Resim seç ve model hangi trafik işareti olduğunu tahmin etsin")

file=st.file_uploader('Bir Resim Sec',type=['jpg','jpeg','png'])

# Eğer dosya yüklenmişse işlemi başlat
if file is not None:
    # Resmi aç ve göster
    img = Image.open(file)
    st.image(img, caption='Yüklenen Resim', use_column_width=True)

    # Görüntüyü işleyip tahmini yap
    image = process_image(img)
    prediction = model.predict(image)
    predicted_class = np.argmax(prediction)

    predicted_class_name = class_names[predicted_class]

    # Tahmin edilen sınıfı ekrana yazdır
    st.write(f'Tahmin edilen trafik işareti: {class_names[predicted_class]}')


