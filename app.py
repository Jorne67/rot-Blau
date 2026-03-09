import streamlit as st
import numpy as np
from keras.models import load_model
from PIL import Image, ImageOps
from supabase import create_client, ClientOptions
import uuid

# -----------------------
# SUPABASE VERBINDUNG
# -----------------------
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]

# Optionen: Auth wird nicht initialisiert, nur Storage/Table
options = ClientOptions(auto_refresh_token=False, persist_session=False)
supabase = create_client(SUPABASE_URL, SUPABASE_KEY, options=options)

BUCKET_NAME = "clothes-images"

# -----------------------
# MODEL LADEN
# -----------------------
np.set_printoptions(suppress=True)
model = load_model("keras_model.h5", compile=False)
class_names = [line.strip() for line in open("labels.txt", "r").readlines()]

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
    class_name = class_names[index]
    confidence_score = prediction[0][index]
    return class_name, confidence_score

# -----------------------
# BILD IN SUPABASE LADEN
# -----------------------
def upload_image(uploaded_file):
    file_name = f"{uuid.uuid4()}.jpg"
    try:
        supabase.storage.from_(BUCKET_NAME).upload(
            file_name,
            uploaded_file.getvalue(),
            content_type="image/jpeg"
        )
    except Exception as e:
        st.error(f"Fehler beim Upload: {e}")
        return None

    image_url = supabase.storage.from_(BUCKET_NAME).get_public_url(file_name)["publicUrl"]
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
            if image_url:
                try:
                    supabase.table("clothes").insert({
                        "class_name": class_name,
                        "image_url": image_url
                    }).execute(throw_on_error=True)
                    st.success("Fundstück gespeichert!")
                except Exception as e:
                    st.error(f"Fehler beim Speichern in Supabase: {e}")

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

        try:
            response = supabase.table("clothes") \
                .select("image_url") \
                .eq("class_name", class_name) \
                .execute(throw_on_error=True)
            results = response.data

            if results:
                st.subheader("Mögliche Matches")
                for item in results:
                    st.image(item["image_url"], width=200)
            else:
                st.warning("Keine passenden Einträge gefunden.")
        except Exception as e:
            st.error(f"Fehler beim Abrufen aus Supabase: {e}")
