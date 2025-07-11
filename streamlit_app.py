import streamlit as st
import subprocess
import time
import requests
from datetime import datetime
from pathlib import Path
import uuid
import os

API_URL = "http://127.0.0.1:8000/convert-to-line-art/"
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)
st.title("Photo-to-Line-Art")

# 尝试启动 FastAPI 服务
def start_fastapi():
    try:
        res = requests.get("http://127.0.0.1:8000/docs")
        if res.status_code == 200:
            st.success("✅ FastAPI is running")
            return
    except:
        st.warning("⚠️ FastAPI is not running, trying to start...")
        subprocess.Popen(["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"])
        time.sleep(5)  # 等待服务启动

start_fastapi()

uploaded_file = st.file_uploader("upload image", type=["jpg", "jpeg", "png"])
# 预览图片
if uploaded_file is not None:
    st.image(uploaded_file, caption=None, use_container_width=True)

controlnet_index = st.selectbox("select model", ["sd-controlnet-canny", "sd-controlnet-depth", "sd-controlnet-hed"])
index_map = {"sd-controlnet-canny": 0, "sd-controlnet-depth": 1, "sd-controlnet-hed": 2}
prompt = st.text_input("Input Prompt", value="clean cartoon-style line art, black and white, clean lines")
low_threshold = st.slider("Low Threshold", 0, 500, 100)
high_threshold = st.slider("High Threshold", 0, 500, 200)

if st.button("Generate Line Art"):
    if uploaded_file is None:
        st.error("Please upload an image first")
    else:
        # Save temporary uploaded image
        image_bytes = uploaded_file.read()
        temp_name = f"temp_{uuid.uuid4().hex[:8]}.png"
        temp_path = f"./{temp_name}"
        with open(temp_path, "wb") as f:
            f.write(image_bytes)

        # Build curl command
        timestamp = datetime.now().strftime("%Y%m%d")
        serial = len(list(OUTPUT_DIR.glob(f"line_art_{timestamp}_*.png"))) + 1
        serial_str = f"{serial:04d}"
        output_name = f"line_art_{timestamp}_{serial_str}.png"

        curl_cmd = [
            "curl", "-X", "POST", API_URL,
            "-F", f"controlnet_index={index_map[controlnet_index]}",
            "-F", f"file=@{temp_path}",
            "-F", f"low_threshold={low_threshold}",
            "-F", f"high_threshold={high_threshold}",
            "-F", f"prompt={prompt}",
            "--output", str(OUTPUT_DIR / output_name)
        ]

        with st.spinner("Running..."):
            result = subprocess.run(curl_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        os.remove(temp_path)

        if (OUTPUT_DIR / output_name).exists():
            st.image(str(OUTPUT_DIR / output_name), caption=output_name)
            st.success("✅ Generated successfully! ")
        else:
            st.error("❌ Generation failed")
            st.text(result.stderr.decode())
