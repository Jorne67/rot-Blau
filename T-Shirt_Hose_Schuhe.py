import streamlit as st
import sqlite3
import numpy as np
from keras.models import load_model
from PIL import Image, ImageOps
import os
from datetime import datetime

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="Katharineum Fundsachen",
    page_icon="👕",
    layout="wide"
)

st.title("👕 Katharineum Fundsachen")
st.markdown("### Digitales Fundbüro")

# ---------------------------------------------------
# MODEL LADEN
# ---------------------------------------------------
np.set_printoptions(suppress=True)
model = load_model("keras_model.h5", compile=False)
class_names = open("labels.txt", "r").readlines()

# ---------------------------------------------------
# DATABASE
# ---------------------------------------------------
conn = sqlite3.connect("clothes.db", check_same_thread=False)
c = conn.cursor()

c.execute("""
CREATE TABLE IF NOT EXISTS clothes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    class_name TEXT,
    image_path TEXT,
    tags TEXT,
    status TEXT DEFAULT 'offen',
    date TEXT
)
""")
conn.commit()

# ---------------------------------------------------
# FUNKTIONEN
# ---------------------------------------------------
def prepare_image(image):
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array
    return data

def predict(image):
    data = prepare_image(image)
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index][2:].strip()
    confidence_score = prediction[0][index]
    return class_name, confidence_score

# ---------------------------------------------------
# SIDEBAR NAVIGATION
# ---------------------------------------------------
menu = st.sidebar.radio(
    "Navigation",
    ["📋 Aktuelle Fundsachen", "➕ Fundstück melden", "🔍 Fundstück suchen"]
)

# ---------------------------------------------------
# ALLE OFFENEN FUNDSACHEN
# ---------------------------------------------------
if menu == "📋 Aktuelle Fundsachen":

    st.subheader("Alle offenen Fundsachen")

    search = st.text_input("🔎 Suche nach Kategorie oder Tag")

    c.execute("""
    SELECT id, class_name, image_path, tags, date
    FROM clothes
    WHERE status='offen'
    ORDER BY id DESC
    """)
    items = c.fetchall()

    filtered = []

    for item in items:
        id, class_name, image_path, tags, date = item
        if search.lower() in class_name.lower() or search.lower() in (tags or "").lower():
            filtered.append(item)

    cols = st.columns(3)

    for index, item in enumerate(filtered):
        id, class_name, image_path, tags, date = item

        with cols[index % 3]:
            st.image(image_path, use_column_width=True)
            st.markdown(f"**{class_name}**")
            st.caption(f"🏷️ {tags}")
            st.caption(f"📅 {date}")

            if st.button("✅ Als abgeholt markieren", key=id):
                c.execute("UPDATE clothes SET status='abgeholt' WHERE id=?", (id,))
                conn.commit()
                st.rerun()

# ---------------------------------------------------
# FUNDSTÜCK MELDEN
# ---------------------------------------------------
elif menu == "➕ Fundstück melden":

    uploaded_file = st.file_uploader("Bild hochladen", type=["jpg", "png", "jpeg"])
    tags_input = st.text_input("Tags (Kommagetrennt, z.B. rot, nike, baumwolle)")

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image)

        class_name, confidence = predict(image)

        st.success(f"Erkannt als: {class_name}")
        st.write(f"Confidence: {confidence:.2f}")

        if st.button("Speichern"):
            os.makedirs("images", exist_ok=True)
            image_path = f"images/{uploaded_file.name}"
            image.save(image_path)

            c.execute("""
            INSERT INTO clothes (class_name, image_path, tags, date)
            VALUES (?, ?, ?, ?)
            """, (
                class_name,
                image_path,
                tags_input.lower(),
                datetime.now().strftime("%d.%m.%Y")
            ))

            conn.commit()
            st.success("Fundstück gespeichert!")

# ---------------------------------------------------
# FUNDSTÜCK SUCHEN
# ---------------------------------------------------
elif menu == "🔍 Fundstück suchen":

    uploaded_file = st.file_uploader("Bild hochladen", type=["jpg", "png", "jpeg"])
    tag_filter = st.text_input("Optional: Nach Tag filtern")

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image)

        class_name, confidence = predict(image)

        st.info(f"Erkannt als: {class_name}")

        c.execute("""
        SELECT image_path, tags
        FROM clothes
        WHERE class_name=? AND status='offen'
        """, (class_name,))

        results = c.fetchall()

        matches = []

        for row in results:
            image_path, tags = row
            if tag_filter.lower() in (tags or ""):
                matches.append(row)

        if matches:
            st.subheader("Mögliche Treffer")
            cols = st.columns(3)
            for i, match in enumerate(matches):
                with cols[i % 3]:
                    st.image(match[0], use_column_width=True)
                    st.caption(f"🏷️ {match[1]}")
        else:
            st.warning("Keine passenden Einträge gefunden.")
