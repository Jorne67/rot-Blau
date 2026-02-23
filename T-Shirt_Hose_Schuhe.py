import streamlit as st
import sqlite3
import numpy as np
from keras.models import load_model
from PIL import Image, ImageOps
import os

# -----------------------
# MODEL LADEN
# -----------------------
np.set_printoptions(suppress=True)
model = load_model("keras_model.h5", compile=False)
class_names = open("labels.txt", "r").readlines()

# -----------------------
# DATABASE SETUP
# -----------------------
conn = sqlite3.connect("clothes.db", check_same_thread=False)
c = conn.cursor()

c.execute("""
CREATE TABLE IF NOT EXISTS clothes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    class_name TEXT,
    image_path TEXT
)
""")
conn.commit()

# -----------------------
# BILD VORBEREITEN
# -----------------------
def prepare_image(image):
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array
    return data

# -----------------------
# VORHERSAGE
# -----------------------
def predict(image):
    data = prepare_image(image)
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index][2:].strip()
    confidence_score = prediction[0][index]
    return class_name, confidence_score

# -----------------------
# STREAMLIT UI
# -----------------------
st.title("👕 Lost & Found Clothes App")

option = st.radio("Was möchtest du tun?", 
                  ["Kleidungsstück melden", "Fundstück suchen"])

# -----------------------
# KLEIDUNG SPEICHERN
# -----------------------
if option == "Kleidungsstück melden":
    uploaded_file = st.file_uploader("Bild hochladen", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Hochgeladenes Bild", use_column_width=True)
        
        class_name, confidence = predict(image)
        
        st.success(f"Erkannt als: {class_name}")
        st.write(f"Confidence: {confidence:.2f}")
        
        if st.button("Speichern"):
            os.makedirs("images", exist_ok=True)
            image_path = f"images/{uploaded_file.name}"
            image.save(image_path)
            
            c.execute("INSERT INTO clothes (class_name, image_path) VALUES (?, ?)",
                      (class_name, image_path))
            conn.commit()
            
            st.success("Gespeichert!")

# -----------------------
# FUNDSTÜCK SUCHEN
# -----------------------
elif option == "Fundstück suchen":
    uploaded_file = st.file_uploader("Fundstück hochladen", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Fundstück", use_column_width=True)
        
        class_name, confidence = predict(image)
        
        st.info(f"Erkannt als: {class_name}")
        
        c.execute("SELECT image_path FROM clothes WHERE class_name = ?", (class_name,))
        results = c.fetchall()
        
        if results:
            st.write("Mögliche Matches:")
            for row in results:
                st.image(row[0], width=200)
        else:
            st.warning("Keine passenden Einträge gefunden.")
