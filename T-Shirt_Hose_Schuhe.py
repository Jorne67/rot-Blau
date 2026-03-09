import streamlit as st
import numpy as np
from keras.models import load_model
from PIL import Image, ImageOps
from supabase import create_client
import uuid

SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

BUCKET_NAME = "clothes-images"

# -----------------------
# MODEL LADEN
# -----------------------
np.set_printoptions(suppress=True)
model = load_model("keras_model.h5", compile=False)
class_names = open("labels.txt", "r").readlines()

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
# BILD ZU SUPABASE UPLOADEN
# -----------------------
def upload_image(uploaded_file):

    file_name = f"{uuid.uuid4()}.jpg"

    supabase.storage.from_(BUCKET_NAME).upload(
        file_name,
        uploaded_file.getvalue()
    )

    image_url = supabase.storage.from_(BUCKET_NAME).get_public_url(file_name)

    return image_url

# -----------------------
# STREAMLIT UI
# -----------------------
st.title("👕 Lost & Found Clothes")

option = st.radio(
    "Was möchtest du tun?",
    ["Kleidungsstück melden", "Fundstück suchen"]
)

# -----------------------
# KLEIDUNG MELDEN
# -----------------------
if option == "Kleidungsstück melden":

    uploaded_file = st.file_uploader(
        "Bild hochladen",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:

        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Hochgeladenes Bild", use_column_width=True)

        class_name, confidence = predict(image)

        st.success(f"Erkannt als: {class_name}")
        st.write(f"Confidence: {confidence:.2f}")

        if st.button("Speichern"):

            image_url = upload_image(uploaded_file)

            supabase.table("clothes").insert({
                "class_name": class_name,
                "image_url": image_url
            }).execute()

            st.success("Fundstück gespeichert!")

# -----------------------
# FUNDSTÜCK SUCHEN
# -----------------------
elif option == "Fundstück suchen":

    uploaded_file = st.file_uploader(
        "Fundstück hochladen",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:

        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Fundstück", use_column_width=True)

        class_name, confidence = predict(image)

        st.info(f"Erkannt als: {class_name}")

        response = supabase.table("clothes") \
            .select("image_url") \
            .eq("class_name", class_name) \
            .execute()

        results = response.data

        if results:

            st.subheader("Mögliche Matches")

            for item in results:
                st.image(item["image_url"], width=200)

        else:
            st.warning("Keine passenden Einträge gefunden.")
