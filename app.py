import streamlit as st
import numpy as np
import cv2
from PIL import Image
from inference import CrowdCounter
from heatmap import create_density_heatmap
import io


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(page_title="Crowd Density Estimation", page_icon="👥", layout="wide")


# ============================================================
# SIMPLE PROFESSIONAL CSS
# ============================================================

st.markdown(
    """
    <style>

    /* Main page */
    .block-container {
        max-width: 1250px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* Header */
    .main-title {
        font-size: 2.3rem;
        font-weight: 700;
        color: #1f2937;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        color: #6b7280;
        font-size: 1rem;
        margin-bottom: 2rem;
    }

    /* Section headings */
    h2, h3 {
        color: #1f2937;
    }

    /* Upload area */
    [data-testid="stFileUploader"] {
        border: 1px solid #e5e7eb;
        border-radius: 10px;
        padding: 10px;
        background-color: #fafafa;
    }

    /* Button */
    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
        height: 45px;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #9ca3af;
        font-size: 0.8rem;
        margin-top: 3rem;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="main-title">👥 Crowd Density Estimation</div>
    <div class="subtitle">
        AI-powered crowd counting and density analysis using P2PNet
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# LOAD MODEL
# ============================================================


@st.cache_resource
def load_model():

    return CrowdCounter("weights/SHTechA.pth")


try:
    model = load_model()

except Exception as e:
    st.error("Model loading failed")

    st.exception(e)

    st.stop()


# ============================================================
# UPLOAD IMAGE
# ============================================================

st.subheader("Upload Crowd Image")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])


# ============================================================
# MAIN APPLICATION
# ============================================================

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    # --------------------------------------------------------
    # IMAGE PREVIEW
    # --------------------------------------------------------

    st.subheader("Image Preview")

    preview_col1, preview_col2, preview_col3 = st.columns([1, 2, 1])

    with preview_col2:
        st.image(image, use_container_width=True)

    st.write("")

    # --------------------------------------------------------
    # ANALYZE BUTTON
    # --------------------------------------------------------

    button_col1, button_col2, button_col3 = st.columns([1, 2, 1])

    with button_col2:
        analyze_button = st.button(
            "🔍 Analyze Crowd", type="primary", use_container_width=True
        )

    # --------------------------------------------------------
    # ANALYSIS
    # --------------------------------------------------------

    if analyze_button:
        with st.spinner("Analyzing crowd image..."):
            result = model.predict(image)

        count = result["count"]

        points = result["points"]

        processed_image = result["image"]

        # ----------------------------------------------------
        # IMAGE PROCESSING
        # ----------------------------------------------------

        image_array = np.array(processed_image)

        height, width = image_array.shape[:2]

        # ----------------------------------------------------
        # HEATMAP
        # ----------------------------------------------------

        heatmap, density = create_density_heatmap(points, width, height)

        # ----------------------------------------------------
        # POINT VISUALIZATION
        # ----------------------------------------------------

        point_image = image_array.copy()

        for point in points:
            x = int(point[0])

            y = int(point[1])

            cv2.circle(point_image, (x, y), 5, (255, 0, 0), -1)

        # ----------------------------------------------------
        # DENSITY LEVEL
        # ----------------------------------------------------

        if count == 0:
            density_level = "No Crowd"

        elif count < 20:
            density_level = "Low Density"

        elif count < 50:
            density_level = "Medium Density"

        elif count < 100:
            density_level = "High Density"

        else:
            density_level = "Very High Density"

        # ====================================================
        # RESULTS
        # ====================================================

        st.divider()

        st.subheader("Analysis Results")

        # ----------------------------------------------------
        # NATIVE STREAMLIT METRICS
        # ----------------------------------------------------

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(label="👥 Estimated People", value=count)

        with col2:
            st.metric(label="📊 Density Level", value=density_level)

        with col3:
            st.metric(label="⚡ Processing Device", value=str(model.device).upper())

        st.write("")

        # ----------------------------------------------------
        # SUMMARY
        # ----------------------------------------------------

        st.info(
            f"The model detected approximately {count} people in the uploaded image."
        )

        # ====================================================
        # VISUAL RESULTS
        # ====================================================

        st.subheader("Visual Analysis")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 📍 Detected Crowd Points")

            st.image(point_image, use_container_width=True)

            st.caption(f"{len(points)} detection points identified")

        with col2:
            st.markdown("#### 🔥 Density Heatmap")

            st.image(heatmap, use_container_width=True)

            st.caption("Brighter regions indicate higher crowd concentration")

        # ====================================================
        # DOWNLOAD
        # ====================================================

        st.subheader("Download Result")

        heatmap_image = Image.fromarray(heatmap)

        buffer = io.BytesIO()

        heatmap_image.save(buffer, format="PNG")

        st.download_button(
            label="⬇️ Download Heatmap",
            data=buffer.getvalue(),
            file_name="crowd_density_heatmap.png",
            mime="image/png",
        )


# ============================================================
# EMPTY STATE
# ============================================================

else:
    st.info("Upload a crowd image to begin the analysis.")


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        Crowd Density Estimation • P2PNet
    </div>
    """,
    unsafe_allow_html=True,
)
