import streamlit as st
from PIL import Image
import subprocess
import time
import os
from streamlit_autorefresh import st_autorefresh

# Paths
IMAGE_DIR = "generated_images"
GESTURE_SCRIPT_PATH = "prediction_art_form_v6.py"
PROMPT_LOG = os.path.join(IMAGE_DIR, "last_prompt.txt")
LOG_FILE = "logs/output.txt"
os.makedirs("logs", exist_ok=True)

st.set_page_config(layout="wide")
st_autorefresh(interval=5000, key="image_refresh")

st.title("✨ Real-time Gesture-Controlled Weather Art Generator")

col1, col2 = st.columns([1, 1])

# Session state for process
if "process" not in st.session_state:
    st.session_state.process = None

with col1:
    st.header("🎯 Gesture Detection")
    st.markdown("Click the button below to launch the OpenCV-based gesture detection with overlays.")

    def launch_script():
        with open(LOG_FILE, "w") as log_file:
            process = subprocess.Popen(
                ["python", GESTURE_SCRIPT_PATH],
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True
            )
            st.session_state.process = process

    if st.button("▶️ Run Real-Time Detection"):
        launch_script()
        st.success("Gesture detection script launched in a separate window.")

    st.markdown("---")
    st.subheader("🖥 Detection Log Output")
    if os.path.exists(LOG_FILE):
        try:
            with open(LOG_FILE, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
                if lines:
                    st.text_area("Live Output", "".join(lines[-20:]), height=300)
                else:
                    st.info("No output yet.")
        except Exception as e:
            st.error(f"Error reading log file: {e}")

    else:
        st.info("Log file not found.")

# --- Auto-refresh image panel ---
with col2:
    st.header("🎨 Last Generated Image")

    # Display last used prompt before image
    if os.path.exists(PROMPT_LOG):
        with open(PROMPT_LOG, "r", encoding="utf-8") as f:
            last_prompt = f.read().strip()
        st.info(last_prompt)

    image_files = sorted(
        [f for f in os.listdir(IMAGE_DIR) if f.endswith(".png")],
        key=lambda f: os.path.getmtime(os.path.join(IMAGE_DIR, f)),
        reverse=True
    )

    if image_files:
        latest_image_path = os.path.join(IMAGE_DIR, image_files[0])
        st.image(Image.open(latest_image_path), caption=image_files[0], use_container_width=True)
    else:
        st.info("No image has been generated yet.")
