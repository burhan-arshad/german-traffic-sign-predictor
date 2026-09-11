import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf


st.set_page_config(
    page_title="Traffic Sign Recognition",
    page_icon="🚦",
    layout="centered",
)

MODEL_PATH = "best_traffic_sign_model.keras"
IMG_SIZE = (48, 48)

CLASS_NAMES = {
    0: "Speed limit (20km/h)",
    1: "Speed limit (30km/h)",
    2: "Speed limit (50km/h)",
    3: "Speed limit (60km/h)",
    4: "Speed limit (70km/h)",
    5: "Speed limit (80km/h)",
    6: "End of speed limit (80km/h)",
    7: "Speed limit (100km/h)",
    8: "Speed limit (120km/h)",
    9: "No passing",
    10: "No passing (vehicles over 3.5 tons)",
    11: "Right-of-way at next intersection",
    12: "Priority road",
    13: "Yield",
    14: "Stop",
    15: "No vehicles",
    16: "Vehicles over 3.5 tons prohibited",
    17: "No entry",
    18: "General caution",
    19: "Dangerous curve (left)",
    20: "Dangerous curve (right)",
    21: "Double curve",
    22: "Bumpy road",
    23: "Slippery road",
    24: "Road narrows on the right",
    25: "Road work",
    26: "Traffic signals",
    27: "Pedestrians",
    28: "Children crossing",
    29: "Bicycles crossing",
    30: "Beware of ice/snow",
    31: "Wild animals crossing",
    32: "End of all speed and passing limits",
    33: "Turn right ahead",
    34: "Turn left ahead",
    35: "Ahead only",
    36: "Go straight or right",
    37: "Go straight or left",
    38: "Keep right",
    39: "Keep left",
    40: "Roundabout mandatory",
    41: "End of no passing",
    42: "End of no passing (vehicles over 3.5 tons)",
}


st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        text-align: center;
        color: #6b7280;
        margin-bottom: 1.8rem;
    }

    .result-card {
        padding: 1.2rem 1.5rem;
        border-radius: 12px;
        background: linear-gradient(135deg, #eff6ff 0%, #f0f9ff 100%);
        border: 1px solid #bfdbfe;
        margin-top: 1rem;
    }

    .result-title {
        font-size: 1.1rem;
        color: #6b7280;
        margin-bottom: 0.2rem;
    }

    .result-class {
        font-size: 1.45rem;
        font-weight: 700;
        color: #1e3a8a;
    }

    .confidence-badge {
        display: inline-block;
        margin-top: 0.5rem;
        padding: 0.25rem 0.75rem;
        border-radius: 999px;
        background: #2563eb;
        color: white;
        font-weight: 600;
        font-size: 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


def preprocess(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB").resize(IMG_SIZE)

    arr = np.array(image).astype("float32")

    arr = np.expand_dims(arr, axis=0)

    return arr


st.markdown(
    '<div class="main-title">🚦 Traffic Sign Recognition</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    'Upload a German traffic sign image and the CNN model will predict '
    'which of the 43 traffic sign categories it belongs to.'
    '</div>',
    unsafe_allow_html=True,
)


try:
    model = load_model()
    model_loaded = True

except Exception as e:
    model_loaded = False

    st.error(
        f"Model file not found or failed to load "
        f"('{MODEL_PATH}'). Make sure the model file is in the same "
        f"folder as app.py.\n\nDetails: {e}"
    )


uploaded_file = st.file_uploader(
    "Upload a traffic sign image",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=False,
)


col1, col2 = st.columns([1, 1])


if uploaded_file is not None:

    image = Image.open(uploaded_file)

    with col1:
        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
        )

    if model_loaded:

        with st.spinner("Analyzing traffic sign..."):

            input_arr = preprocess(image)

            predictions = model.predict(
                input_arr,
                verbose=0
            )[0]

            top_idx = int(np.argmax(predictions))

            confidence = float(
                predictions[top_idx]
            ) * 100

            top5_idx = np.argsort(
                predictions
            )[::-1][:5]

        with col2:

            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-title">Predicted Sign</div>
                    <div class="result-class">
                        {CLASS_NAMES.get(top_idx, f"Class {top_idx}")}
                    </div>
                    <span class="confidence-badge">
                        {confidence:.1f}% confidence
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("#### Top 5 Predictions")

        for idx in top5_idx:

            label = CLASS_NAMES.get(
                int(idx),
                f"Class {idx}"
            )

            percentage = float(
                predictions[idx]
            ) * 100

            st.write(
                f"**{label}** — {percentage:.1f}%"
            )

            st.progress(
                min(int(percentage), 100)
            )

else:

    with col1:
        st.info("👆 Upload an image to get started.")


st.markdown("---")

st.caption(
    "Model: CNN (Conv2D 32→64→128) trained on GTSRB · "
    "43 classes · Built by CypherAI"
)