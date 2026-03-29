import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess

st.set_page_config(
    page_title="AI vs Real Image Classifier",
    page_icon="🧠",
    layout="centered"
)

@st.cache_resource
def load_models():
    cnn = load_model("models/cifake_model.keras")
    eff = load_model("models/efficientnet_cifake.keras")
    return cnn, eff

cnn_model, efficient_model = load_models()

st.title("🧠 AI vs Real Image Classifier")
st.markdown("Upload an image and classify it using **Custom CNN** or **EfficientNetB0**.")

with st.sidebar:
    st.header("⚙️ Settings")
    model_choice = st.selectbox(
        "Choose a model",
        ["Custom CNN", "EfficientNetB0"]
    )

    threshold = st.slider(
        "Classification threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.01
    )

    if model_choice == "Custom CNN":
        selected_model = cnn_model
        target_size = (128, 128)
        st.info("Custom CNN selected\n\nInput size: 128 × 128\n\nPreprocessing: divide by 255")
    else:
        selected_model = efficient_model
        target_size = (224, 224)
        st.info("EfficientNetB0 selected\n\nInput size: 224 × 224\n\nPreprocessing: EfficientNet preprocess_input")

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

def preprocess_image(image: Image.Image, model_name: str) -> np.ndarray:
    image = image.convert("RGB")

    if model_name == "Custom CNN":
        image = image.resize((128, 128))
        img_array = np.array(image, dtype=np.float32) / 255.0
    else:
        image = image.resize((224, 224))
        img_array = np.array(image, dtype=np.float32)
        img_array = efficientnet_preprocess(img_array)

    img_array = np.expand_dims(img_array, axis=0)
    return img_array

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)

    with col2:
        st.markdown("### Selected Configuration")
        st.write(f"**Model:** {model_choice}")
        st.write(f"**Resize:** {target_size[0]} × {target_size[1]}")
        st.write(f"**Threshold:** {threshold:.2f}")
        st.write("**Class Mapping:** FAKE = High Score, REAL = Low Score")

    img_array = preprocess_image(image, model_choice)

    with st.spinner("Predicting..."):
        raw_prediction = float(selected_model.predict(img_array, verbose=0)[0][0])

    raw_prediction = max(0.0, min(raw_prediction, 1.0))

    # --- THE FIX IS HERE ---
    # Because your model is "inverted", the raw prediction is the FAKE score.
    fake_score = raw_prediction
    real_score = 1.0 - raw_prediction

    st.markdown("---")
    st.subheader("📊 Prediction Result")

    # Applying the working threshold logic: score <= threshold means REAL
    if raw_prediction <= threshold:
        st.success("✅ Prediction: Real Image")
        confidence = real_score
        predicted_label = "Real"
    else:
        st.error("❌ Prediction: AI-Generated Image")
        confidence = fake_score
        predicted_label = "AI-Generated"

    st.write(f"**Predicted Class:** {predicted_label}")
    st.write(f"**Confidence:** {confidence:.2%}")

    st.markdown("### Probability Scores")
    st.write(f"**Real:** {real_score:.2%}")
    st.progress(real_score)

    st.write(f"**AI-Generated:** {fake_score:.2%}")
    st.progress(fake_score)

    st.markdown("### Raw Output")
    st.code(
        f"Raw model output : {raw_prediction:.6f}\n"
        f"Real score       : {real_score:.6f}\n"
        f"AI score         : {fake_score:.6f}",
        language="text"
    )

else:
    st.info("Please upload an image to start prediction.")
